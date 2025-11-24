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
    uint64_t        nnz; 
} Var_Entry;

typedef struct _Laik_Layout_Var Laik_Layout_Var;
struct _Laik_Layout_Var {
    Laik_Layout h;
    Var_Entry* e;
};


//--------------------------------------------------------------
// interface implementation of variable layout
//

uint64_t laik_variable_layout_map_nnz(Laik_Layout* l, int mapNo)
{
    Laik_Layout_Var* vl = (Laik_Layout_Var*) l;
    assert(mapNo >= 0 && mapNo < l->map_count);
    return vl->e[mapNo].nnz;
}

// forward decl
static int64_t offset_variable(Laik_Layout* l, int n, Laik_Index* idx);

// return lex layout if given layout is a lexicographical layout
Laik_Layout_Var* laik_is_layout_variable(Laik_Layout* l)
{
    if (l->offset == offset_variable)
        return (Laik_Layout_Var*) l;

    return 0; // not a lexicographical layout
}

// ...existing code...
static int section_variable(Laik_Layout* l, Laik_Index* idx) {
    // 1D: determine which map (section) contains global index idx->i[0].
    Laik_Layout_Var* vl = (Laik_Layout_Var*) l;
    int64_t g = idx->i[0];

    for (int sec = 0; sec < l->map_count; ++sec) {
        int64_t start = vl->e[sec].first_boundary;
        int64_t end;

        if (sec < l->map_count - 1) {
            // Next map’s first_boundary is the exclusive end of this one
            end = vl->e[sec + 1].first_boundary;
        }
        else {
            // Last map: reconstruct end from its own prefix slice length
            // count = (to - from)   OR (to - from + 1) depending on construction rule.
            // We stored first_boundary = from; original 'to' = from + (count) for last map.
            end = start + (int64_t)vl->e[sec].count;
        }

        if (g >= start && g < end)
            return sec;
    }
    return -1; // not found
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
    Var_Entry* e = &vl->e[sec];
    if (!e->row_ptr || e->count < 2) return 0;
    int64_t li = idx->i[0] - e->first_boundary;
    if (li == (int64_t)(e->count - 1)) // sentinel (exclusive end)
        return e->row_ptr[li] - e->row_ptr[0];
    assert(li >= 0 && li < (int64_t)(e->count - 1));
    return e->row_ptr[li] - e->row_ptr[0];
}


extern int laik_layout_pack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);
extern char* laik_layout_describe_gen(Laik_Layout* l);
extern int laik_layout_unpack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);

// ...existing code...
Laik_Layout* laik_new_layout_variable(int n, Laik_Range* ranges, Laik_Data_Parameters* params)
{
    assert(params && params->prefix_row_data);
    Laik_Data* prefix = params->prefix_row_data;
    assert(prefix->activeMappings);

    int64_t* rp_local = 0;
    uint64_t rp_len  = 0;
    laik_get_map_1d(prefix, 0, (void**)&rp_local, &rp_len);
    assert(rp_local && rp_len > 0);

    Laik_Layout_Var* vl = (Laik_Layout_Var*) malloc(sizeof(Laik_Layout_Var));
    assert(vl);

    laik_init_layout(&vl->h, 1, n, 0,
                     section_variable,
                     mapno_variable,
                     offset_variable,
                     reuse_variable,
                     laik_layout_describe_gen,
                     laik_layout_pack_gen,
                     laik_layout_unpack_gen,
                     laik_layout_copy_gen);

    vl->e = (Var_Entry*) calloc(n, sizeof(Var_Entry));
    assert(vl->e);

    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        int64_t from = ranges[i].from.i[0];
        int64_t to   = ranges[i].to.i[0];
        assert(to > from);

        uint64_t boundary_count = (uint64_t)(to - from + 1);

        vl->e[i].count          = boundary_count;
        vl->e[i].row_ptr        = rp_local;
        vl->e[i].first_boundary = from;

        uint64_t nnz_i = (uint64_t)(rp_local[boundary_count - 1] - rp_local[0]);
        total += nnz_i;
        vl->e[i].nnz = nnz_i;
    }
    vl->h.count = total;

    return (Laik_Layout*) vl;
}