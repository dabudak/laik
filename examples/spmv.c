 
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

Laik_Layout*     laik_new_layout_variable(int n, Laik_Range* ranges, Laik_Data_Parameters* params);
Laik_Partitioner* laik_new_var_block_partitioner1(void);

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

    laik_set_phase(inst, 0, "init", NULL);

    // rowD is CSR row-pointer(prefix)
    Laik_Space* prefix_space = laik_new_space_1d(inst, size + 1);
    Laik_Data*  rowD = laik_new_data(prefix_space, laik_Int64);

    // valD / colD live on "rows" space (size). variable layout will map rows->nnz.
    Laik_Space* rows_space = laik_new_space_1d(inst, size);
    Laik_Data*  valD = laik_new_data(rows_space, laik_Double);
    Laik_Data*  colD = laik_new_data(rows_space, laik_Int64);

    // keep everything on master initially so master process can initialize.
    Laik_Partitioner* master_pr = laik_new_master_partitioner();

    Laik_Partitioning* row_master_p = laik_new_partitioning(master_pr, world, prefix_space, 0);

    laik_switchto_partitioning(rowD, row_master_p, LAIK_DF_None, LAIK_RO_None);

    // Attach parameters (prefix_row_data) BEFORE setting variable layout factories
    Laik_Data_Parameters* vparams = (Laik_Data_Parameters*)malloc(sizeof(*vparams));
    vparams->prefix_row_data = rowD;

    // Now set variable layout factory (will consume params in prepareMaps)
    laik_data_set_layout_factory(valD, laik_new_layout_variable);
    laik_data_set_layout_factory(colD, laik_new_layout_variable);

    // Master partitioning for valD / colD
    Laik_Partitioning* val_master_p = laik_new_partitioning(master_pr, world, rows_space, 0);
    Laik_Partitioning* col_master_p = laik_new_partitioning(master_pr, world, rows_space, 0);

    int64_t* rp_dbg = NULL; uint64_t rl_dbg = 0;
    laik_get_map_1d(rowD, 0, (void**)&rp_dbg, &rl_dbg);
    int64_t fr=0,tr=0; laik_my_range_1d(row_master_p,0,&fr,&tr);

    // master initializes row-pointer, then val/col
    if (rank == 0) {
        // fill row prefix
        int64_t* rp = NULL; uint64_t rp_len = 0;
        laik_get_map_1d(rowD, 0, (void**)&rp, &rp_len);
        assert(rp_len == (uint64_t)(size + 1));

        int off = 0;
        for (int r = 0; r < size; ++r) {
            rp[r] = off;
            off  += r;
        }
        rp[size] = off;  // total nnz
    }

    laik_data_attach_params(valD, vparams);
    laik_data_attach_params(colD, vparams);

    laik_switchto_partitioning(valD, val_master_p, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(colD, col_master_p, LAIK_DF_None, LAIK_RO_None);

    if (rank == 0) {
        int64_t* col = NULL; uint64_t col_len = 0;
        double*  val = NULL; uint64_t val_len = 0;
        laik_get_map_1d(colD, 0, (void**)&col, &col_len);
        laik_get_map_1d(valD, 0, (void**)&val, &val_len);

        int64_t* rp = NULL; uint64_t rp_len = 0;
        laik_get_map_1d(rowD, 0, (void**)&rp, &rp_len);
        const int nnz = (int)rp[size];
        assert((int)col_len == nnz && (int)val_len == nnz);

        int off = 0;
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < r; ++c) {
                col[off] = c;
                val[off] = (double)(size - r);
                ++off;
            }
        }
        assert(off == nnz);
    }



    // repartition everything for computation

    Laik_Partitioner* rows_var_pr = laik_new_var_block_partitioner1();
    Laik_Partitioning* row_var_p  = laik_new_partitioning(rows_var_pr, world, prefix_space, 0);

    laik_switchto_partitioning(rowD, row_var_p, LAIK_DF_Preserve, LAIK_RO_None);

    Laik_Partitioner* blk_pr      = laik_new_block_partitioner1();
    Laik_Partitioning* val_blk_p  = laik_new_partitioning(blk_pr, world, rows_space, 0);
    Laik_Partitioning* col_blk_p  = laik_new_partitioning(blk_pr, world, rows_space, 0);

    laik_switchto_partitioning(valD, val_blk_p, LAIK_DF_Preserve, LAIK_RO_None);
    laik_switchto_partitioning(colD, col_blk_p, LAIK_DF_Preserve, LAIK_RO_None);

    double* v = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; ++i) v[i] = (double)(i + 1);

    Laik_Data*        resD      = laik_new_data(rows_space, laik_Double);
    Laik_Partitioning* res_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    laik_switchto_partitioning(resD, res_blk_p, LAIK_DF_None, LAIK_RO_None);

    // SPMV #1: compute + gather to master
    laik_set_phase(inst, 1, "1st SpmV", NULL);

    double*  res    = NULL;
    uint64_t rcount = 0;
    laik_get_map_1d(resD, 0, (void**)&res, &rcount);
    for (uint64_t i = 0; i < rcount; ++i) res[i] = 0.0;

    int64_t fromRow = 0, toRow = 0;
    laik_my_range_1d(res_blk_p, 0, &fromRow, &toRow);

    // local slices
    const int64_t* row_ptr = NULL; uint64_t rp_len = 0;
    int64_t*       col     = NULL; uint64_t col_len = 0;
    double*        val     = NULL; uint64_t val_len = 0;
    
    laik_get_map_1d(rowD, 0, (void**)&row_ptr, &rp_len);
    laik_get_map_1d(colD, 0, (void**)&col,     &col_len);
    laik_get_map_1d(valD, 0, (void**)&val,     &val_len);

    const int64_t base = row_ptr[0];           // global prefix offset of first local row
    assert(base <= row_ptr[rp_len - 1]);

    for (int r = (int)fromRow; r < (int)toRow; ++r) {
        const int lr      = r - (int)fromRow;
        const int64_t beg = row_ptr[lr]     - base;
        const int64_t end = row_ptr[lr + 1] - base;

        // Debug: nnz in this row should be r
        int64_t nnz_row = end - beg;
        if (nnz_row != r) {
            fprintf(stderr, "DEBUG rank=%d row=%d nnz_row=%lld expected=%d "
                            "rp[lr]=%lld rp[lr+1]=%lld base=%lld\n",
                    rank, r, (long long)nnz_row, r,
                    (long long)row_ptr[lr], (long long)row_ptr[lr+1], (long long)base);
            assert(nnz_row == r && "row_ptr slice mismatch");
        }

        for (int64_t o = beg; o < end; ++o)
            res[lr] += val[o] * v[col[o]];
        laik_set_iteration(inst, lr);
    }

    // Verify locally (element-wise)
    {
        const double eps = 1e-9;
        for (int r = (int)fromRow; r < (int)toRow; ++r) {
            int lr = r - (int)fromRow;
            double expected = (double)(size - r) * 0.5 * (double)r * (double)(r + 1);
            if (fabs(res[lr] - expected) > eps) {
                fprintf(stderr, "Mismatch rank=%d row=%d got=%g exp=%g\n",
                        rank, r, res[lr], expected);
                assert(0 && "SPMV check failed");
            }
        }
    }

    Laik_Partitioning* pMaster = laik_new_partitioning(laik_Master, world, rows_space, 0);
    laik_switchto_partitioning(resD, pMaster, LAIK_DF_Preserve, LAIK_RO_None);
    if (rank == 0) {
        laik_get_map_1d(resD, 0, (void**)&res, &rcount);
        double sum = 0.0;
        for (uint64_t i = 0; i < rcount; ++i) sum += res[i];
        // Verify global sum against analytic formula
        long double expectedSum = 0.0L;
        for (int r = 0; r < size; ++r) {
            expectedSum += (long double)(size - r) * 0.5L * (long double)r * (long double)(r + 1);
        }
        printf("Res sum (regular): %f (expected: %.0Lf)\n", sum, expectedSum);
        assert(fabsl((long double)sum - expectedSum) < 1e-6L);
    }


    // SPMV #2: reduction path
    laik_iter_reset(inst);
    laik_set_phase(inst, 2, "2nd SpmV", NULL);

    Laik_Partitioning* pAll = laik_new_partitioning(laik_All, world, rows_space, 0);
    laik_switchto_partitioning(resD, pAll, LAIK_DF_Init, LAIK_RO_Sum);

    laik_get_map_1d(resD, 0, (void**)&res, &rcount);
    laik_my_range_1d(res_blk_p, 0, &fromRow, &toRow);

    laik_get_map_1d(rowD, 0, (void**)&row_ptr, &rp_len);
    laik_get_map_1d(colD, 0, (void**)&col, &col_len);
    laik_get_map_1d(valD, 0, (void**)&val, &val_len);

    const int64_t base2 = row_ptr[0];
    for (int r = (int)fromRow; r < (int)toRow; ++r) {
        const int lr      = r - (int)fromRow;
        const int64_t beg = row_ptr[lr]     - base2;
        const int64_t end = row_ptr[lr + 1] - base2;
        for (int64_t o = beg; o < end; ++o)
            res[lr] += val[o] * v[col[o]];
        laik_set_iteration(inst, lr);
    }

    laik_switchto_partitioning(resD, pMaster, LAIK_DF_Preserve, LAIK_RO_Sum);
    if (rank == 0) {
        laik_get_map_1d(resD, 0, (void**)&res, &rcount);
        double sum = 0.0;
        for (uint64_t i = 0; i < rcount; ++i) sum += res[i];
        printf("Res sum (reduce): %f\n", sum);
    }

    laik_finalize(inst);
    free(vparams);
    return 0;
}