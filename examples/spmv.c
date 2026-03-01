 
/* This file is part of the LAIK parallel container library.
 * Copyright (c) 2017 Josef Weidendorfer
 *
 * LAIK is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 3.
 *
 * LAIK is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * SPMV example.
 */

#include <laik.h>
#include <laik/data.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static double prefix_weight_ignore_last(Laik_Index* idx, const void* userData)
{
    const int64_t last = *((const int64_t*)userData);
    return (idx->i[0] == last) ? 0.0 : 1.0;
}

// maximal size
#define MAXSIZE 10000

int main(int argc, char* argv[])
{
    Laik_Instance* inst  = laik_init(&argc, &argv);
    Laik_Group*    world = laik_world(inst);
    int rank = laik_myid(world);

    int size = 0;
    if (argc > 1) size = atoi(argv[1]);
    if ((size <= 0) || (size > MAXSIZE)) size = MAXSIZE;

    double* v = (double*) malloc(size * sizeof(double));
    assert(v);
    for (int i = 0; i < size; ++i)
        v[i] = (double)(i + 1);

    laik_set_phase(inst, 0, "init", NULL);

    // rowD is CSR row-pointer(prefix)
    Laik_Space* prefix_space = laik_new_space_1d(inst, size + 1);

    // valD / colD live on "rows" space (size). variable layout will map rows->nnz.
    Laik_Space* rows_space = laik_new_space_1d(inst, size);

    // 1) Create data objects
    Laik_Data*  rowD = laik_new_data(prefix_space, laik_Int64);
    Laik_Data*  valD = laik_new_data(rows_space,   laik_Double);
    Laik_Data*  colD = laik_new_data(rows_space,   laik_Int64);
    Laik_Data*  resD = laik_new_data(rows_space,   laik_Double);

    // 2) Build full prefix (master)
    Laik_Partitioner* master_pr   = laik_new_master_partitioner();
    Laik_Partitioning* row_master = laik_new_partitioning(master_pr, world, prefix_space, 0);
    laik_switchto_partitioning(rowD, row_master, LAIK_DF_None, LAIK_RO_None);

    if (rank == 0) {
        int64_t* rp = NULL; uint64_t rp_len = 0;
        laik_get_map_1d(rowD, 0, (void**)&rp, &rp_len);
        assert(rp_len == size + 1);
        int64_t off = 0;
        for (int r = 0; r < size; ++r) {
            rp[r] = off;
            off += r;               // nnz in row r = r
        }
        rp[size] = off;
    }
    Laik_Data_Parameters* vparams = (Laik_Data_Parameters*)malloc(sizeof(*vparams));
    vparams->prefix_row_data = rowD;
    vparams->var_ranges = 0;
    vparams->var_range_count = 0;
    laik_data_attach_params(valD, vparams);
    laik_data_attach_params(colD, vparams);
    laik_data_set_layout_factory(valD, laik_new_layout_variable);
    laik_data_set_layout_factory(colD, laik_new_layout_variable);

    Laik_Partitioning* val_master_p = laik_new_partitioning(master_pr, world, rows_space, 0);
    Laik_Partitioning* col_master_p = laik_new_partitioning(master_pr, world, rows_space, 0);
    laik_switchto_partitioning(valD, val_master_p, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(colD, col_master_p, LAIK_DF_None, LAIK_RO_None);

    if (rank == 0) {
        int64_t* rp = NULL; uint64_t rp_len = 0;
        laik_get_map_1d(rowD, 0, (void**)&rp, &rp_len);
        assert(rp_len == (uint64_t)(size + 1));
        int64_t total_nnz = rp[size];

        double*  val = NULL; uint64_t val_len = 0;
        int64_t* col = NULL; uint64_t col_len = 0;
        laik_get_map_1d(valD, 0, (void**)&val, &val_len);
        laik_get_map_1d(colD, 0, (void**)&col, &col_len);
        assert((int64_t)val_len == total_nnz && (int64_t)col_len == total_nnz);

        int64_t off = 0;
        for (int64_t r = 0; r < size; ++r) {
            for (int64_t c = 0; c < r; ++c) {
                col[off] = c;
                val[off] = (double)(size - r);
                ++off;
            }
        }
        assert(off == total_nnz);
    }

    
    // Prefix space has size+1 elements (row boundaries). We want cut positions to
    // match the row partitioning on [0;size[, so ignore the final boundary index
    // in the weighting while still keeping it in the last range.
    int64_t last_boundary = (int64_t)size;
    Laik_Partitioner* rows_var_pr = laik_new_var_block_partitioner(0, 1,
                                                                  prefix_weight_ignore_last,
                                                                  0, &last_boundary);
    Laik_Partitioning* row_var_p  = laik_new_partitioning(rows_var_pr, world, prefix_space, 0);
    laik_switchto_partitioning(rowD, row_var_p, LAIK_DF_Preserve, LAIK_RO_None);

    Laik_Partitioner* blk_pr     = laik_new_block_partitioner(0, 1, 0, 0, 0);
    Laik_Partitioning* val_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    Laik_Partitioning* col_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    laik_switchto_partitioning(valD, val_blk_p, LAIK_DF_Preserve, LAIK_RO_None);
    laik_switchto_partitioning(colD, col_blk_p, LAIK_DF_Preserve, LAIK_RO_None);


    // 7) Partition result
    Laik_Partitioning* res_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    laik_switchto_partitioning(resD, res_blk_p, LAIK_DF_None, LAIK_RO_None);

    // 8) SpMV
    laik_set_phase(inst, 1, "SpMV", NULL);
    laik_iter_reset(inst);

    {
        int mapCount = laik_my_mapcount(res_blk_p);
        for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
            int mrCount = laik_my_maprangecount(res_blk_p, mapNo);
            assert(mrCount == 1);
            Laik_TaskRange* tr = laik_my_maprange(res_blk_p, mapNo, 0);
            const Laik_Range* s = laik_taskrange_get_range(tr);
            int64_t fromRow = s->from.i[0];
            int64_t toRow   = s->to.i[0];

            // row_ptr slice for boundaries [fromRow..toRow]
            int rm = -1;
            uint64_t lfrom_u = 0;
            Laik_Mapping* pm = laik_global2maplocal_1d(rowD, fromRow, &rm, &lfrom_u);
            assert(pm);
            const int64_t* rp_base = NULL; uint64_t rp_len = 0;
            laik_get_map_1d(rowD, rm, (void**)&rp_base, &rp_len);
            assert(lfrom_u < rp_len);
            assert((uint64_t)(toRow - fromRow + 1) <= (rp_len - lfrom_u));
            const int64_t* row_ptr = rp_base + (int64_t) lfrom_u;

            double*  val = NULL; uint64_t val_len = 0;
            int64_t* col = NULL; uint64_t col_len = 0;
            laik_get_map_1d(valD, mapNo, (void**)&val, &val_len);
            laik_get_map_1d(colD, mapNo, (void**)&col, &col_len);
            assert(val_len == col_len);

            double* res_map = NULL; uint64_t rcount = 0;
            laik_get_map_1d(resD, mapNo, (void**)&res_map, &rcount);

            int resMapNo = -1;
            uint64_t res_lfrom_u = 0;
            Laik_Mapping* resm = laik_global2maplocal_1d(resD, fromRow, &resMapNo, &res_lfrom_u);
            assert(resm);
            assert(resMapNo == mapNo);
            assert(res_lfrom_u < rcount);
            double* res = res_map + (int64_t) res_lfrom_u;

            uint64_t rowsHere = (uint64_t)(toRow - fromRow);
            assert(rcount - res_lfrom_u >= rowsHere);

            for (uint64_t i = 0; i < rowsHere; ++i) res[i] = 0.0;

            int64_t base = row_ptr[0];
            for (int64_t r = fromRow; r < toRow; ++r) {
                int lr = (int)(r - fromRow);
                int64_t beg = row_ptr[lr]     - base;
                int64_t end = row_ptr[lr + 1] - base;
                for (int64_t o = beg; o < end; ++o)
                    res[lr] += val[o] * v[col[o]];
                laik_set_iteration(inst, (int)r);
            }
        }
    }

    // 9) Gather & reduce
    Laik_Partitioning* pMaster = laik_new_partitioning(laik_Master, world, rows_space, 0);
    laik_switchto_partitioning(resD, pMaster, LAIK_DF_Preserve, LAIK_RO_None);
    if (rank == 0) {
        double* res=NULL; uint64_t rcount=0;
        laik_get_map_1d(resD,0,(void**)&res,&rcount);
        double sum=0.0;
        for (uint64_t i=0;i<rcount;++i) sum+=res[i];
        printf("Res sum (regular): %f\n", sum);
    }

    laik_switchto_partitioning(resD, pMaster, LAIK_DF_Preserve, LAIK_RO_Sum);
    if (rank == 0) {
        double* res=NULL; uint64_t rcount=0;
        laik_get_map_1d(resD,0,(void**)&res,&rcount);
        double sum=0.0;
        for (uint64_t i=0;i<rcount;++i) sum+=res[i];
        printf("Res sum (reduce): %f\n", sum);
    }

    free(v);
    free(vparams);
    laik_finalize(inst);
    return 0;
}