/*
 * Robust compact vector layout for LAIK.
 * Maps locally owned indices to [0..localLength-1] and external indices to
 * [localLength..localLength+externalCount-1] using a stable global->local map.
 */

#include "laik-internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _Laik_Vector_Pair {
    int64_t global;
    int64_t local;
} Laik_Vector_Pair;

typedef struct _Laik_Layout_Vector Laik_Layout_Vector;
struct _Laik_Layout_Vector {
    Laik_Layout h;
    uint64_t localLength;
    uint64_t numberOfExternalValues;
    uint64_t totalLength;
    uint64_t mapSize;
    Laik_Vector_Pair map[]; // sorted by global index
};

// forward decls
static int section_vector(Laik_Layout* l, Laik_Index* idx);
static int mapno_vector(Laik_Layout* l, int n);
static int64_t offset_vector(Laik_Layout* l, int n, Laik_Index* idx);
static bool reuse_vector(Laik_Layout* l, int n, Laik_Layout* o, int no);
static char* describe_vector(Laik_Layout* l);
static unsigned int pack_vector(Laik_Mapping* m, Laik_Range* range, Laik_Index* idx, char* buf, unsigned int size);
static unsigned int unpack_vector(Laik_Mapping* m, Laik_Range* range, Laik_Index* idx, char* buf, unsigned int size);
static void copy_vector(Laik_Range* range, Laik_Mapping* from, Laik_Mapping* to);
static uint64_t size_vector(Laik_Layout* l, int n, Laik_Index* idx);

static int cmp_pair(const void* a, const void* b)
{
    const Laik_Vector_Pair* pa = (const Laik_Vector_Pair*)a;
    const Laik_Vector_Pair* pb = (const Laik_Vector_Pair*)b;
    if (pa->global < pb->global) return -1;
    if (pa->global > pb->global) return 1;
    return 0;
}

Laik_Layout_Vector* laik_is_layout_vector(Laik_Layout* l)
{
    if (l->offset == offset_vector)
        return (Laik_Layout_Vector*) l;
    return 0;
}

uint64_t laik_vector_layout_total(Laik_Layout* l)
{
    Laik_Layout_Vector* lv = laik_is_layout_vector(l);
    if (!lv) return 0;
    return lv->totalLength;
}

static int section_vector(Laik_Layout* l, Laik_Index* idx)
{
    Laik_Layout_Vector* lv = laik_is_layout_vector(l);
    assert(lv);
    (void)idx;
    return (lv->totalLength > 0) ? 0 : -1;
}

static int mapno_vector(Laik_Layout* l, int n)
{
    assert(n < l->map_count);
    assert(n == 0);
    return n;
}

static int64_t offset_vector(Laik_Layout* l, int n, Laik_Index* idx)
{
    assert(n == 0);
    Laik_Layout_Vector* lv = laik_is_layout_vector(l);
    assert(lv);
    int64_t key = idx->i[0];
    int64_t lo = 0;
    int64_t hi = (int64_t)lv->mapSize - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int64_t g = lv->map[mid].global;
        if (g == key) return lv->map[mid].local;
        if (g < key) lo = mid + 1; else hi = mid - 1;
    }
    {
        char msg[128];
        snprintf(msg, sizeof(msg), "vector layout: global index %lld not in map", (long long)key);
        laik_panic(msg);
        exit(1);
    }
}

static bool next_idx(Laik_Range* range, Laik_Index* idx)
{
    idx->i[0]++;
    if (idx->i[0] < range->to.i[0])
        return true;
    return false;
}

static bool reuse_vector(Laik_Layout* l, int n, Laik_Layout* o, int no)
{
    (void)l; (void)n; (void)o; (void)no;
    return false;
}

static char* describe_vector(Laik_Layout* l)
{
    static char s[200];
    Laik_Layout_Vector* lv = laik_is_layout_vector(l);
    assert(lv);
    snprintf(s, sizeof(s), "vector (local=%llu, external=%llu, total=%llu)",
             (unsigned long long)lv->localLength,
             (unsigned long long)lv->numberOfExternalValues,
             (unsigned long long)lv->totalLength);
    return s;
}

static unsigned int pack_vector(Laik_Mapping* m, Laik_Range* range, Laik_Index* idx,
                                char* buf, unsigned int size)
{
    unsigned int elemsize = m->data->elemsize;
    Laik_Layout* layout = m->layout;
    if (laik_index_isEqual(layout->dims, idx, &(range->to)))
        return 0;

    assert(laik_range_within_range(range, &(m->requiredRange)));

    unsigned int count = 0;
    while (size >= elemsize) {
        int64_t off = layout->offset(layout, m->layoutSection, idx);
        void* idxPtr = m->start + off * elemsize;
        memcpy(buf, idxPtr, elemsize);
        size -= elemsize;
        buf += elemsize;
        count++;
        if (!next_idx(range, idx)) {
            *idx = range->to;
            break;
        }
    }
    return count;
}

static unsigned int unpack_vector(Laik_Mapping* m, Laik_Range* range, Laik_Index* idx,
                                  char* buf, unsigned int size)
{
    unsigned int elemsize = m->data->elemsize;
    Laik_Layout* layout = m->layout;
    assert(size > 0);
    assert(!laik_index_isEqual(layout->dims, idx, &(range->to)));
    assert(laik_range_within_range(range, &(m->requiredRange)));

    unsigned int count = 0;
    while (size >= elemsize) {
        int64_t off = layout->offset(layout, m->layoutSection, idx);
        void* idxPtr = m->start + off * elemsize;
        memcpy(idxPtr, buf, elemsize);
        size -= elemsize;
        buf += elemsize;
        count++;
        if (!next_idx(range, idx)) {
            *idx = range->to;
            break;
        }
    }
    return count;
}

static void copy_vector(Laik_Range* range, Laik_Mapping* from, Laik_Mapping* to)
{
    unsigned int elemsize = from->data->elemsize;
    assert(elemsize == to->data->elemsize);

    Laik_Index idx = range->from;
    uint64_t count = 0;
    do {
        int64_t fromOff = from->layout->offset(from->layout, from->layoutSection, &idx);
        int64_t toOff = to->layout->offset(to->layout, to->layoutSection, &idx);
        void* fromPtr = from->start + fromOff * elemsize;
        void* toPtr = to->start + toOff * elemsize;
        memcpy(toPtr, fromPtr, elemsize);
        count++;
    } while (next_idx(range, &idx));
    assert(count == laik_range_size(range));
}

static uint64_t size_vector(Laik_Layout* l, int n, Laik_Index* idx)
{
    (void)l;
    (void)n;
    (void)idx;
    return 1;
}

Laik_Layout* laik_new_layout_vector(int n, Laik_Range* ranges, Laik_Data_Parameters* params)
{
    assert(n == 1); // compact vector layout supports one mapping
    assert(params);
    assert(params->vector_local_indices || params->vector_local_count == 0);
    assert(params->vector_external_indices || params->vector_external_count == 0);

    uint64_t total = params->vector_local_count + params->vector_external_count;
    size_t bytes = sizeof(Laik_Layout_Vector) + total * sizeof(Laik_Vector_Pair);
    Laik_Layout_Vector* lv = (Laik_Layout_Vector*)malloc(bytes);
    if (!lv) {
        laik_panic("Out of memory allocating Laik_Layout_Vector");
        exit(1);
    }

    lv->localLength = params->vector_local_count;
    lv->numberOfExternalValues = params->vector_external_count;
    lv->totalLength = lv->localLength + lv->numberOfExternalValues;
    lv->mapSize = lv->totalLength;

    if (lv->totalLength > 0) {
        uint64_t pos = 0;
        for (uint64_t i = 0; i < lv->localLength; ++i) {
            lv->map[pos].global = params->vector_local_indices[i];
            lv->map[pos].local = (int64_t)i;
            pos++;
        }
        for (uint64_t i = 0; i < lv->numberOfExternalValues; ++i) {
            lv->map[pos].global = params->vector_external_indices[i];
            lv->map[pos].local = (int64_t)(lv->localLength + i);
            pos++;
        }
        qsort(lv->map, lv->mapSize, sizeof(Laik_Vector_Pair), cmp_pair);
    }

    laik_init_layout(&(lv->h), 1, n, lv->totalLength,
                     section_vector,
                     mapno_vector,
                     offset_vector,
                     reuse_vector,
                     describe_vector,
                     pack_vector,
                     unpack_vector,
                     copy_vector,
                     size_vector);

    (void)ranges; // ranges not needed for stable mapping
    return (Laik_Layout*)lv;
}
