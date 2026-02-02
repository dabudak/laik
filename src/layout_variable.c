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
#include <stdlib.h>
#include <string.h>

typedef struct _Var_Entry {
    uint64_t        count;
    const int64_t*  row_ptr;         // private copy of boundaries for this section
    int64_t         first_boundary;  // global boundary index for row_ptr[0]
    uint64_t        nnz; 
    uint64_t        base_off;        // element offset from mapping->start for this section
} Var_Entry;

typedef struct _Laik_Layout_Var Laik_Layout_Var;
struct _Laik_Layout_Var {
    Laik_Layout h;
    Var_Entry e[];
};


//--------------------------------------------------------------
// interface implementation of variable layout
//


// forward decls
static int64_t offset_variable(Laik_Layout* l, int n, Laik_Index* idx);
static int section_variable(Laik_Layout* l, Laik_Index* idx);
static int  mapno_variable(Laik_Layout* l, int n);
static bool reuse_variable(Laik_Layout* l, int n, Laik_Layout* o, int no);

// return lex layout if given layout is a lexicographical layout
Laik_Layout_Var* laik_is_layout_variable(Laik_Layout* l)
{
    if (l->offset == offset_variable)
        return (Laik_Layout_Var*) l;

    return 0; // not a lexicographical layout
}

uint64_t laik_variable_layout_map_nnz(Laik_Layout* l, int mapNo)
{
    Laik_Layout_Var* vl = (Laik_Layout_Var*) l;
    assert(mapNo >= 0 && mapNo < l->map_count);
    return vl->e[mapNo].nnz;
}


static char* describe_variable(Laik_Layout* l) {
    (void)l;
    return (char*)"variable";
}

// pack rows, returning number of logical elements (rows) packed
static
unsigned int pack_variable(Laik_Mapping* m, Laik_Range* range,
                           Laik_Index* idx, char* buf, unsigned int size)
{
    Laik_Layout_Var* vl = (Laik_Layout_Var*) m->layout;
    Var_Entry* e = &vl->e[m->layoutSection];

    int64_t g = idx->i[0];
    int64_t g_to = range->to.i[0];
    int64_t first = e->first_boundary;

    unsigned int used_bytes = 0;
    unsigned int packed_rows = 0;
    while (g < g_to) {
        int64_t li = g - first;
        if (li < 0 || li >= (int64_t)(e->count - 1)) break;
        uint64_t nnz_row = (uint64_t)(e->row_ptr[li + 1] - e->row_ptr[li]);
        uint64_t bytes_row = nnz_row * m->data->elemsize;
        if ((uint64_t)used_bytes + bytes_row > (uint64_t)size) break;
        int64_t off = (int64_t)e->base_off + (int64_t)(e->row_ptr[li] - e->row_ptr[0]);
        memcpy(buf + used_bytes, m->start + off * m->data->elemsize, bytes_row);
        used_bytes += (unsigned int)bytes_row;
        packed_rows++;
        g++;
    }

    idx->i[0] = g;
    return packed_rows;
}

// unpack rows, returning number of logical elements (rows) unpacked
static
unsigned int unpack_variable(Laik_Mapping* m, Laik_Range* range,
                             Laik_Index* idx, char* buf, unsigned int size)
{
    Laik_Layout_Var* vl = (Laik_Layout_Var*) m->layout;
    Var_Entry* e = &vl->e[m->layoutSection];

    int64_t g = idx->i[0];
    int64_t g_to = range->to.i[0];
    int64_t first = e->first_boundary;

    unsigned int used_bytes = 0;
    unsigned int unpacked_rows = 0;
    while (g < g_to) {
        int64_t li = g - first;
        if (li < 0 || li >= (int64_t)(e->count - 1)) break;
        uint64_t nnz_row = (uint64_t)(e->row_ptr[li + 1] - e->row_ptr[li]);
        uint64_t bytes_row = nnz_row * m->data->elemsize;
        if ((uint64_t)used_bytes + bytes_row > (uint64_t)size) break;
        int64_t off = (int64_t)e->base_off + (int64_t)(e->row_ptr[li] - e->row_ptr[0]);
        memcpy(m->start + off * m->data->elemsize, buf + used_bytes, bytes_row);
        used_bytes += (unsigned int)bytes_row;
        unpacked_rows++;
        g++;
    }

    idx->i[0] = g;
    return unpacked_rows;
}

// copy rows range (variable span)
static
void copy_variable(Laik_Range* range, Laik_Mapping* from, Laik_Mapping* to)
{
    Laik_Layout_Var* vlf = (Laik_Layout_Var*) from->layout;
    Var_Entry* ef = &vlf->e[from->layoutSection];
    Laik_Layout_Var* vlt = (Laik_Layout_Var*) to->layout;
    Var_Entry* et = &vlt->e[to->layoutSection];

    int64_t g_from = range->from.i[0];
    int64_t g_to   = range->to.i[0];
    while (g_from < g_to) {
        int64_t li_f = g_from - ef->first_boundary;
        int64_t li_t = g_from - et->first_boundary;
        uint64_t nnz_row = (uint64_t)(ef->row_ptr[li_f + 1] - ef->row_ptr[li_f]);
        int64_t off_src = (int64_t)ef->base_off + (int64_t)(ef->row_ptr[li_f] - ef->row_ptr[0]);
        int64_t off_dst = (int64_t)et->base_off + (int64_t)(et->row_ptr[li_t] - et->row_ptr[0]);
        uint64_t bytes = nnz_row * from->data->elemsize;
        memcpy(to->start + off_dst * to->data->elemsize,
               from->start + off_src * from->data->elemsize,
               bytes);
        g_from++;
    }
}

static int section_variable(Laik_Layout* l, Laik_Index* idx) {
    Laik_Layout_Var* vl = (Laik_Layout_Var*) l;
    int64_t g = idx->i[0];
    // Rows are addressed in a half-open range [from,to).
    // Each section stores (rows+1) boundaries, so the last boundary index
    // equals the exclusive end row index and must not be treated as a row.
    for (int sec = 0; sec < l->map_count; ++sec) {
        int64_t start = vl->e[sec].first_boundary;
        int64_t end_excl = start + (int64_t)vl->e[sec].count - 1;
        if (g >= start && g < end_excl) return sec;
    }

    // Allow the global end sentinel (== end_excl of the last section).
    int last_sec = -1;
    int64_t last_end_excl = 0;
    for (int sec = 0; sec < l->map_count; ++sec) {
        int64_t start = vl->e[sec].first_boundary;
        int64_t end_excl = start + (int64_t)vl->e[sec].count - 1;
        if (last_sec < 0 || end_excl > last_end_excl) {
            last_sec = sec;
            last_end_excl = end_excl;
        }
    }
    if (g == last_end_excl) return last_sec;

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
    Var_Entry* e = &vl->e[sec];
    if (!e->row_ptr) return 0;
    int64_t li = idx->i[0] - e->first_boundary;
    if (li == (int64_t)(e->count - 1)) // sentinel (exclusive end)
        return (int64_t)e->base_off + (e->row_ptr[li] - e->row_ptr[0]);
    assert(li >= 0 && li < (int64_t)(e->count - 1));
    return (int64_t)e->base_off + (e->row_ptr[li] - e->row_ptr[0]);
}
// Return element count for logical index in variable layout (rows as elements)
// Size(idx) = row_ptr[li+1] - row_ptr[li], where li = idx->i[0] - first_boundary
static uint64_t size_variable(Laik_Layout* l, int n, Laik_Index* idx)
{
    Laik_Layout_Var* vl = laik_is_layout_variable(l);
    assert(vl);
    assert(n >= 0 && n < l->map_count);
    Var_Entry* e = &vl->e[n];
    assert(e->row_ptr);
    int64_t li = idx->i[0] - e->first_boundary;
    // boundaries array has count = rows + 1 entries; valid rows are [0, count-2]
    assert(li >= 0 && li < (int64_t)(e->count - 1));
    return (uint64_t)(e->row_ptr[li+1] - e->row_ptr[li]);
}

extern int laik_layout_pack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);
extern char* laik_layout_describe_gen(Laik_Layout* l);
extern int laik_layout_unpack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);

Laik_Layout* laik_new_layout_variable(int n, Laik_Range* ranges, Laik_Data_Parameters* params)
{
    assert(params && params->prefix_row_data);
    Laik_Data* prefix = params->prefix_row_data;
    assert(prefix->activeMappings);

    uint64_t total_boundaries = 0;
    for (int i = 0; i < n; ++i) {
        int64_t from = ranges[i].from.i[0];
        int64_t to   = ranges[i].to.i[0];
        assert(to > from);
        total_boundaries += (uint64_t)(to - from + 1);
    }

    size_t bytes = sizeof(Laik_Layout_Var) + ((size_t)n) * sizeof(Var_Entry) + ((size_t)total_boundaries) * sizeof(int64_t);
    Laik_Layout_Var* vl = (Laik_Layout_Var*) malloc(bytes);
    assert(vl);

    laik_init_layout(&vl->h, 1, n, 0,
                     section_variable,
                     mapno_variable,
                     offset_variable,
                     reuse_variable,
                     describe_variable,
                     pack_variable,          // variable-specific pack
                     unpack_variable,        // variable-specific unpack
                     copy_variable, // generic copy;
                     size_variable);  

    memset(vl->e, 0, ((size_t)n) * sizeof(Var_Entry));

    int64_t* boundary_store = (int64_t*) (vl->e + n);
    uint64_t boundary_off = 0;

    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        int64_t from = ranges[i].from.i[0];
        int64_t to   = ranges[i].to.i[0];
        assert(to > from);

        // prefix_row_data is distributed and can have multiple mappings (cycles>1).
        // Find the mapping covering the needed boundaries [from..to] (inclusive).
        int prefix_mapNo = -1;
        uint64_t lfrom_u = 0;
        Laik_Mapping* pm = laik_global2maplocal_1d(prefix, from, &prefix_mapNo, &lfrom_u);
        if (!pm) {
            fprintf(stderr,
                "LAIK variable layout: prefix_row_data does not map required boundary index %lld (needed for rows [%lld;%lld[).\n"
                "This usually means the prefix (row_ptr) partitioning does not cover the same row boundaries as the row partitioning.\n",
                (long long)from, (long long)from, (long long)to);
            abort();
        }
        assert(prefix_mapNo >= 0);

        uint64_t boundary_count = (uint64_t)(to - from + 1);
        vl->e[i].count          = boundary_count;
        vl->e[i].first_boundary = from;
        vl->e[i].row_ptr        = boundary_store + boundary_off;
        vl->e[i].base_off        = total;

        // copy needed boundaries so the layout remains valid even if prefix_row_data repartitions
        for (uint64_t j = 0; j < boundary_count; ++j) {
            int64_t g = from + (int64_t)j;
            int prefix_mapNo2 = -1;
            uint64_t lo_u2 = 0;
            Laik_Mapping* pm2 = laik_global2maplocal_1d(prefix, g, &prefix_mapNo2, &lo_u2);
            assert(pm2);
            int64_t* rp_local2 = 0;
            uint64_t rp_len2 = 0;
            laik_get_map_1d(prefix, prefix_mapNo2, (void**)&rp_local2, &rp_len2);
            assert(rp_local2);
            assert(lo_u2 < rp_len2);
            boundary_store[boundary_off + j] = rp_local2[lo_u2];
        }

        uint64_t nnz_i = (uint64_t)(boundary_store[boundary_off + boundary_count - 1] - boundary_store[boundary_off]);
        vl->e[i].nnz = nnz_i;

        boundary_off += boundary_count;

        total += nnz_i;
    }
    vl->h.count = total;

    return (Laik_Layout*) vl;
}