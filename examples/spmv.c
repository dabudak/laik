 
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

    // 3) Distribute prefix to final row-variable partition (Preserve)
    Laik_Partitioner* rows_var_pr = laik_new_var_block_partitioner1();
    Laik_Partitioning* row_var_p  = laik_new_partitioning(rows_var_pr, world, prefix_space, 0);
    laik_switchto_partitioning(rowD, row_var_p, LAIK_DF_Preserve, LAIK_RO_None);

        // Verify local row_ptr slice correctness
    {
        int64_t rf=0, rt=0;
        laik_my_range_1d(row_var_p,0,&rf,&rt);
        int64_t* rp=NULL; uint64_t rp_len=0;
        laik_get_map_1d(rowD,0,(void**)&rp,&rp_len);
        assert(rp_len == (uint64_t)(rt - rf));
        for (int64_t i=0; i < (int64_t)rp_len; ++i) {
            int64_t g = rf + i;
            int64_t expected = g * (g - 1) / 2;
            if (rp[i] != expected) {
                fprintf(stderr,"PREFIX MISMATCH rank=%d global_row=%lld rp=%lld exp=%lld\n",
                        rank,(long long)g,(long long)rp[i],(long long)expected);
                assert(0);
            }
        }
    }

    // 4) Attach params AFTER rowD is in final distribution
    Laik_Data_Parameters* vparams = (Laik_Data_Parameters*)malloc(sizeof(*vparams));
    vparams->prefix_row_data = rowD;
    laik_data_attach_params(valD, vparams);
    laik_data_attach_params(colD, vparams);
    laik_data_set_layout_factory(valD, laik_new_layout_variable);
    laik_data_set_layout_factory(colD, laik_new_layout_variable);

    // 5) Partition val/col to block; variable layout now sees correct local prefix
    Laik_Partitioner* blk_pr     = laik_new_block_partitioner1();
    Laik_Partitioning* val_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    Laik_Partitioning* col_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    laik_switchto_partitioning(valD, val_blk_p, LAIK_DF_None, LAIK_RO_None);
    laik_switchto_partitioning(colD, col_blk_p, LAIK_DF_None, LAIK_RO_None);

    // 6) Initialize val/col only on ranks that own rows now (no rebuild later needed)
    {
        int64_t fromRow=0,toRow=0;
        laik_my_range_1d(val_blk_p,0,&fromRow,&toRow);
        const int64_t* rp=NULL; uint64_t rp_len=0;
        laik_get_map_1d(rowD,0,(void**)&rp,&rp_len);
        assert(rp_len == (uint64_t)(toRow - fromRow + 1));

        double*  val=NULL; uint64_t val_len=0;
        int64_t* col=NULL; uint64_t col_len=0;
        laik_get_map_1d(valD,0,(void**)&val,&val_len);
        laik_get_map_1d(colD,0,(void**)&col,&col_len);

        int64_t base = rp[0];
        int64_t nnz_local = rp[rp_len-1] - base;
        assert((int64_t)val_len == nnz_local && (int64_t)col_len == nnz_local);

        int64_t off = 0;
        for (int64_t r = fromRow; r < toRow; ++r) {
            for (int64_t c = 0; c < r; ++c) {
                col[off] = c;
                val[off] = (double)(size - r);
                ++off;
            }
        }
        assert(off == nnz_local);
    }

    // 7) Partition result
    Laik_Partitioning* res_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    laik_switchto_partitioning(resD, res_blk_p, LAIK_DF_None, LAIK_RO_None);

    // 8) SpMV
    laik_set_phase(inst, 1, "SpMV", NULL);
    laik_iter_reset(inst);

    {
        int64_t fromRow=0,toRow=0;
        laik_my_range_1d(res_blk_p,0,&fromRow,&toRow);

        const int64_t* row_ptr=NULL; uint64_t rp_len=0;
        laik_get_map_1d(rowD,0,(void**)&row_ptr,&rp_len);
        double* val=NULL; uint64_t val_len=0;
        int64_t* col=NULL; uint64_t col_len=0;
        laik_get_map_1d(valD,0,(void**)&val,&val_len);
        laik_get_map_1d(colD,0,(void**)&col,&col_len);
        double* res=NULL; uint64_t rcount=0;
        laik_get_map_1d(resD,0,(void**)&res,&rcount);

        assert(rcount == (uint64_t)(toRow - fromRow));
        assert(rp_len == (uint64_t)(toRow - fromRow + 1));

        for (uint64_t i=0;i<rcount;++i) res[i]=0.0;

        int64_t base = row_ptr[0];
        for (int64_t r=fromRow; r<toRow; ++r) {
            int lr = (int)(r - fromRow);
            int64_t beg = row_ptr[lr]     - base;
            int64_t end = row_ptr[lr + 1] - base;
            int64_t nnz_row = end - beg;
            if (nnz_row != r) {
                fprintf(stderr,"NNZ MISMATCH rank=%d row=%lld nnz=%lld exp=%lld\n",
                        rank,(long long)r,(long long)nnz_row,(long long)r);
                assert(0);
            }
        }

        for (int64_t r=fromRow; r<toRow; ++r) {
            int lr = (int)(r - fromRow);
            int64_t beg = row_ptr[lr]     - base;
            int64_t end = row_ptr[lr + 1] - base;
            for (int64_t o=beg; o<end; ++o)
                res[lr] += val[o] * v[col[o]];
            laik_set_iteration(inst, lr);
        }
        for (int64_t r=fromRow; r<toRow; ++r) {
            int lr = (int)(r - fromRow);
            double expected = (double)(size - r) * 0.5 * (double)r * (double)(r + 1);
            if (fabs(res[lr] - expected) > 1e-9) {
                fprintf(stderr,"RES MISMATCH rank=%d row=%lld got=%g exp=%g\n",
                        rank,(long long)r,res[lr],expected);
                assert(0);
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
// ...existing code...
}