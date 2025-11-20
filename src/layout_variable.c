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

// helper: nnz in row (li -> li+1)
static inline uint64_t var_row_nnz(const Var_Entry* e, int64_t li)
{
    return (uint64_t)(e->row_ptr[li+1] - e->row_ptr[li]);
}

// pack exactly one row (or as many as fit) treating each logical element as a row
static
unsigned int pack_variable(Laik_Mapping* m, Laik_Range* range,
                           Laik_Index* idx, char* buf, unsigned int size)
{
    Laik_Layout_Var* vl = (Laik_Layout_Var*) m->layout;
    Var_Entry* e = &vl->e[m->layoutSection];

    int64_t g_from = range->from.i[0];
    int64_t g_to   = range->to.i[0];          // exclusive row end
    int64_t first  = e->first_boundary;
    int64_t li_from = g_from - first;
    int64_t li_to   = g_to   - first;

    unsigned int used = 0;
    while (li_from < li_to) {
        uint64_t nnz_row = var_row_nnz(e, li_from);
        uint64_t bytes_row = nnz_row * m->data->elemsize;
        if (used + bytes_row > size) break;   // not enough buffer for next row
        int64_t off = (int64_t)(e->row_ptr[li_from] - e->row_ptr[0]);
        memcpy(buf + used, m->start + off * m->data->elemsize, bytes_row);
        used += (unsigned int)bytes_row;
        li_from++;
    }

    // advance idx to global row after last packed row
    idx->i[0] = first + li_from;
    // return number of rows packed
    return (unsigned int)(li_from - (g_from - first));
}

// unpack rows (mirror of pack)
static
unsigned int unpack_variable(Laik_Mapping* m, Laik_Range* range,
                             Laik_Index* idx, char* buf, unsigned int size)
{
    Laik_Layout_Var* vl = (Laik_Layout_Var*) m->layout;
    Var_Entry* e = &vl->e[m->layoutSection];

    int64_t g_from = range->from.i[0];
    int64_t g_to   = range->to.i[0];
    int64_t first  = e->first_boundary;
    int64_t li_from = g_from - first;
    int64_t li_to   = g_to   - first;

    unsigned int used = 0;
    while (li_from < li_to) {
        uint64_t nnz_row = var_row_nnz(e, li_from);
        uint64_t bytes_row = nnz_row * m->data->elemsize;
        if (used + bytes_row > size) break;
        int64_t off = (int64_t)(e->row_ptr[li_from] - e->row_ptr[0]);
        memcpy(m->start + off * m->data->elemsize, buf + used, bytes_row);
        used += (unsigned int)bytes_row;
        li_from++;
    }

    idx->i[0] = first + li_from;
    return (unsigned int)(li_from - (g_from - first));
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
    int64_t li_f = g_from - ef->first_boundary;
    int64_t li_t = g_from - et->first_boundary;
    while (g_from < g_to) {
        uint64_t nnz_row = (uint64_t)(ef->row_ptr[li_f+1] - ef->row_ptr[li_f]);
        int64_t off_src = (int64_t)(ef->row_ptr[li_f] - ef->row_ptr[0]);
        int64_t off_dst = (int64_t)(et->row_ptr[li_t] - et->row_ptr[0]);
        uint64_t bytes = nnz_row * from->data->elemsize;
        memcpy(to->start + off_dst * to->data->elemsize,
               from->start + off_src * from->data->elemsize,
               bytes);
        g_from++;
        li_f++; li_t++;
    }
}

static int section_variable(Laik_Layout* l, Laik_Index* idx) {
    Laik_Layout_Var* vl = (Laik_Layout_Var*) l;
    int64_t g = idx->i[0];
    for (int sec = 0; sec < l->map_count; ++sec) {
        int64_t start = vl->e[sec].first_boundary;
        int64_t end   = (sec < l->map_count - 1)
                      ? vl->e[sec + 1].first_boundary
                      : start + (int64_t)vl->e[sec].count - 1; // rows = count-1
        if (g >= start && g < end) return sec;
    }
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
        return e->row_ptr[li] - e->row_ptr[0];
    assert(li >= 0 && li < (int64_t)(e->count - 1));
    return e->row_ptr[li] - e->row_ptr[0];
}


extern int laik_layout_pack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);
extern char* laik_layout_describe_gen(Laik_Layout* l);
extern int laik_layout_unpack_gen(Laik_Mapping* m, Laik_Range* r, Laik_Index* idx, char* buf, unsigned int size);

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
                     describe_variable,
                     pack_variable,          // variable-specific pack
                     unpack_variable,        // variable-specific unpack
                     copy_variable);  // generic copy;

    vl->e = (Var_Entry*) calloc(n, sizeof(Var_Entry));
    assert(vl->e);

    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        int64_t from = ranges[i].from.i[0];
        int64_t to   = ranges[i].to.i[0];
        assert(to > from);

        uint64_t boundary_count = (uint64_t)(to - from + 1);
         fprintf(stderr,
                "DEBUG laik_new_layout_variable: i=%d from=%lld to=%lld boundary_count=%llu rp_local=%p rp_local[0]=%lld\n",
                i, (long long)from, (long long)to,
                (unsigned long long)boundary_count,
                (void*)rp_local,
                (long long)rp_local[0]);
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