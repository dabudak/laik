/*
 * This file is part of the LAIK library.
 * Copyright (c) 2017, 2018 Josef Weidendorfer <Josef.Weidendorfer@gmx.de>
 *
 * LAIK is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 3 or later.
 *
 * LAIK is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "laik-internal.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct _Var_Entry {
    uint64_t        count;
    const int64_t*  row_ptr;         // local slice into global-accumulated prefix array
    int64_t         first_boundary;  // global boundary index for row_ptr[0]
} Var_Entry;

typedef struct _Laik_Layout_Var Laik_Layout_Var;
struct _Laik_Layout_Var {
    Laik_Layout h;
    Var_Entry e[0];
};


//--------------------------------------------------------------
// interface implementation of variable layout
//

// forward decl
static int64_t offset_variable(Laik_Layout* l, int n, Laik_Index* idx);

// return lex layout if given layout is a lexicographical layout
static
Laik_Layout_Var* laik_is_layout_variable(Laik_Layout* l)
{
    if (l->offset == offset_variable)
        return (Laik_Layout_Var*) l;

    return 0; // not a lexicographical layout
}

static int section_variable(Laik_Layout* l, Laik_Index* idx) { 
    (void)l;
    (void)idx; 
    return -1; 

}
static int mapno_variable(Laik_Layout* l, int n) { 
    assert(n>=0 && n<l->map_count); 
    return n; 
}

static bool reuse_variable(Laik_Layout* l, int n, Laik_Layout* o, int no){ 
    (void)l;
    (void)n;
    (void)o;
    (void)no; 
    return false; 
}



// return offset for <idx> in map <n> of this layout
// offset = (row_ptr[li] - row_ptr[0]), where li = idx->i[0] - first_boundary
static int64_t offset_variable(Laik_Layout* l, int sec, Laik_Index* idx)
{
    Laik_Layout_Var* vl = (Laik_Layout_Var*) l;
    assert(sec >= 0 && sec < l->map_count);
    Var_Entry* e = &vl->e[sec];

    assert(e->row_ptr && e->count > 0);

    int64_t li = idx->i[0] - e->first_boundary; // turn global boundary into local boundary idx
    assert(li >= 0 && li < (int64_t)e->count);

    return (int64_t)(e->row_ptr[li] - e->row_ptr[0]); // elements (nnz) from start of this map
}


extern int laik_layout_pack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);
extern char* laik_layout_describe_gen(Laik_Layout* l);
extern int laik_layout_unpack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);

Laik_Layout* laik_new_layout_variable(int n, Laik_Range* ranges)
{
    int dims = ranges->space->dims;
    assert(dims == 1);

    Laik_Layout_Var* vl = (Laik_Layout_Var*)
        malloc(sizeof(Laik_Layout_Var) + n * sizeof(Var_Entry));
    if (!vl) {
        laik_panic("Out of memory allocating Laik_Layout_Var");
        exit(1);
    }

    laik_init_layout(&(vl->h), dims, n, 0,
                     section_variable,
                     mapno_variable,
                     offset_variable,
                     reuse_variable,
                     laik_layout_describe_gen,
                     laik_layout_pack_gen,
                     laik_layout_unpack_gen,
                     laik_layout_copy_gen);

    // Initialize entries to a safe empty state.
    for (int i = 0; i < n; ++i) {
        vl->e[i].count = 0;
        vl->e[i].row_ptr = NULL;
        vl->e[i].first_boundary = 0;
    }
    vl->h.count = 0; // will be set in attach

    return (Laik_Layout*) vl;
}

// fills Var_Entry for the current mappings of owner with rowD.
// assumes partitioning is already set and layout created by prepareMaps.
void laik_layout_variable_attach(Laik_Data* owner, Laik_Data* row_data)
{
    assert(owner && row_data);
    assert(owner->activeMappings && "owner must have active mappings");

    Laik_MappingList* ml = owner->activeMappings;
    Laik_Layout* l  = ml->layout;

    // verify that the current layout is our variable layout
    assert(laik_is_layout_variable(l));

    // map row_data locally: must hold global accumulated nnz (rows+1)
    void* rp_base = NULL;
    uint64_t rp_len       = 0;
    laik_get_map_1d(row_data, 0, &rp_base, &rp_len);
    const int64_t* rp = (const int64_t*) rp_base;

    Laik_Layout_Var* vl = (Laik_Layout_Var*) l;

    uint64_t total_elems = 0;
    for (int mapNo = 0; mapNo < ml->count; ++mapNo) {
        Laik_Mapping* m = &ml->map[mapNo];
        const Laik_Range* r = &m->requiredRange;

        int64_t from = r->from.i[0];   // glbal boundary start for this map
        int64_t to   = r->to.i[0];     // global boundary end (exclusiv)

        // "+1 except for the last map" rule
        uint64_t boundary_count = (uint64_t)(to - from);
        if (mapNo < ml->count - 1) boundary_count += 1;
        assert(boundary_count > 0);

        // safety: row_ptr must have at least up to 'to' boundary locally visible
        assert((uint64_t)to <= rp_len && "row_ptr slice out of local map");

        // fill entry
        vl->e[mapNo].count          = boundary_count;
        vl->e[mapNo].row_ptr        = rp;
        vl->e[mapNo].first_boundary = from;

        // nnz for this map = last - first (prefix sums difference)
        uint64_t nnz_i = (uint64_t)(vl->e[mapNo].row_ptr[boundary_count - 1] - vl->e[mapNo].row_ptr[0]);
        total_elems += nnz_i;

    }

    vl->h.count = total_elems;
}
