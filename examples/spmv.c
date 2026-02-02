 
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

static int64_t expected_rowptr(int64_t row)
{
    // rp[row] = sum_{k=0}^{row-1} k = row*(row-1)/2
    return row * (row - 1) / 2;
}

static void debug_check_migrated_slices(const char* label,
                                       int rank,
                                       int size,
                                       Laik_Data* rowD,
                                       Laik_Partitioning* rowP,
                                       Laik_Data* valD,
                                       Laik_Data* colD,
                                       Laik_Partitioning* rowsP,
                                       int max_rows_to_check,
                                       int max_nnz_to_print)
{
    fprintf(stderr, "\n=== MIGRATION CHECK (%s) rank=%d ===\n", label, rank);

    // 1) Check rowD content for all local maps.
    {
        int mapCount = laik_my_mapcount(rowP);
        for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
            int mrCount = laik_my_maprangecount(rowP, mapNo);
            if (mrCount != 1) {
                fprintf(stderr, "rowD: rank=%d map=%d has %d ranges (expected 1)\n",
                        rank, mapNo, mrCount);
            }
            for (int rix = 0; rix < mrCount; ++rix) {
                Laik_TaskRange* tr = laik_my_maprange(rowP, mapNo, rix);
                const Laik_Range* s = laik_taskrange_get_range(tr);
                int64_t rf = s->from.i[0];
                int64_t rt = s->to.i[0];

                int rm = -1;
                uint64_t lfrom_u = 0;
                Laik_Mapping* pm = laik_global2maplocal_1d(rowD, rf, &rm, &lfrom_u);
                if (!pm) {
                    fprintf(stderr, "rowD: rank=%d map=%d range=[%lld,%lld) missing boundary %lld\n",
                            rank, mapNo, (long long)rf, (long long)rt, (long long)rf);
                    continue;
                }

                int64_t* rp_base = NULL; uint64_t rp_len = 0;
                laik_get_map_1d(rowD, rm, (void**)&rp_base, &rp_len);
                if (lfrom_u >= rp_len) {
                    fprintf(stderr, "rowD: rank=%d map=%d lfrom=%llu >= rp_len=%llu\n",
                            rank, mapNo,
                            (unsigned long long)lfrom_u,
                            (unsigned long long)rp_len);
                    continue;
                }

                int64_t* rp = rp_base + (int64_t)lfrom_u;
                int64_t n = rt - rf;
                if (n > (int64_t)(rp_len - lfrom_u)) n = (int64_t)(rp_len - lfrom_u);
                int64_t show = n;
                if (show > max_rows_to_check) show = max_rows_to_check;

                fprintf(stderr, "rowD: rank=%d map=%d global=[%lld,%lld) local_len=%llu show=%lld\n",
                        rank, mapNo, (long long)rf, (long long)rt,
                        (unsigned long long)rp_len, (long long)show);

                for (int64_t i = 0; i < show; ++i) {
                    int64_t g = rf + i;
                    int64_t got = rp[i];
                    int64_t exp = expected_rowptr(g);
                    if (got != exp) {
                        fprintf(stderr,
                                "rowD MISMATCH: rank=%d map=%d g=%lld got=%lld exp=%lld (rm=%d lfrom=%llu)\n",
                                rank, mapNo, (long long)g, (long long)got, (long long)exp,
                                rm, (unsigned long long)lfrom_u);
                        break;
                    }
                }
            }
        }
    }

    // 2) Check valD/colD content for all local maps.
    if (!valD || !colD || !rowsP) return;

    {
        int mapCount = laik_my_mapcount(rowsP);
        for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
            int mrCount = laik_my_maprangecount(rowsP, mapNo);
            if (mrCount != 1) {
                fprintf(stderr, "rows: rank=%d map=%d has %d ranges (expected 1)\n",
                        rank, mapNo, mrCount);
            }
            for (int rix = 0; rix < mrCount; ++rix) {
                Laik_TaskRange* tr = laik_my_maprange(rowsP, mapNo, rix);
                const Laik_Range* s = laik_taskrange_get_range(tr);
                int64_t rowFrom = s->from.i[0];
                int64_t rowTo   = s->to.i[0];

                // get row_ptr slice for boundaries [rowFrom..rowTo]
                int rm = -1;
                uint64_t lfrom_u = 0;
                Laik_Mapping* pm = laik_global2maplocal_1d(rowD, rowFrom, &rm, &lfrom_u);
                if (!pm) {
                    fprintf(stderr, "val/col: rank=%d map=%d missing row_ptr boundary at rowFrom=%lld\n",
                            rank, mapNo, (long long)rowFrom);
                    continue;
                }
                const int64_t* rp_base = NULL; uint64_t rp_len = 0;
                laik_get_map_1d(rowD, rm, (void**)&rp_base, &rp_len);
                if (lfrom_u >= rp_len) {
                    fprintf(stderr, "val/col: rank=%d map=%d row_ptr lfrom=%llu >= rp_len=%llu\n",
                            rank, mapNo,
                            (unsigned long long)lfrom_u,
                            (unsigned long long)rp_len);
                    continue;
                }
                const int64_t* row_ptr = rp_base + (int64_t)lfrom_u;

                double*  val = NULL; uint64_t val_len = 0;
                int64_t* col = NULL; uint64_t col_len = 0;
                laik_get_map_1d(valD, mapNo, (void**)&val, &val_len);
                laik_get_map_1d(colD, mapNo, (void**)&col, &col_len);

                uint64_t rowsHere = (uint64_t)(rowTo - rowFrom);
                if (rowsHere + 1 > (rp_len - lfrom_u)) {
                    fprintf(stderr,
                            "val/col: rank=%d map=%d row_ptr slice too short rowsHere=%llu rpAvail=%llu\n",
                            rank, mapNo,
                            (unsigned long long)rowsHere,
                            (unsigned long long)(rp_len - lfrom_u));
                    continue;
                }

                int64_t base = row_ptr[0];
                int64_t local_nnz_expected = row_ptr[rowsHere] - base;

                // quick checks
                fprintf(stderr,
                        "val/col: rank=%d map=%d rows=[%lld,%lld) rowsHere=%llu nnzExp=%lld val_len=%llu col_len=%llu (row_ptr_map=%d)\n",
                        rank, mapNo, (long long)rowFrom, (long long)rowTo,
                        (unsigned long long)rowsHere, (long long)local_nnz_expected,
                        (unsigned long long)val_len, (unsigned long long)col_len, rm);

                if ((int64_t)val_len != local_nnz_expected || (int64_t)col_len != local_nnz_expected) {
                    fprintf(stderr,
                            "LENGTH MISMATCH: rank=%d map=%d nnzExp=%lld val_len=%llu col_len=%llu\n",
                            rank, mapNo, (long long)local_nnz_expected,
                            (unsigned long long)val_len, (unsigned long long)col_len);
                }

                // compute and print small sample; also compute simple checksums
                double val_sum = 0.0;
                long double col_sum = 0.0;
                int64_t sampleRows = (int64_t)rowsHere;
                if (sampleRows > max_rows_to_check) sampleRows = max_rows_to_check;

                for (int64_t li = 0; li < (int64_t)rowsHere && li < (int64_t)val_len; ++li) {
                    (void)li;
                }
                for (uint64_t o = 0; o < val_len; ++o) {
                    val_sum += val[o];
                    col_sum += (long double)col[o];
                }

                fprintf(stderr,
                        "checksums: rank=%d map=%d val_sum=%.3f col_sum=%.0Lf\n",
                        rank, mapNo, val_sum, col_sum);

                for (int64_t lr = 0; lr < sampleRows; ++lr) {
                    int64_t r = rowFrom + lr;
                    int64_t beg = row_ptr[lr]     - base;
                    int64_t end = row_ptr[lr + 1] - base;
                    int64_t nnz = end - beg;
                    fprintf(stderr, "ROW %lld nnz=%lld :", (long long)r, (long long)nnz);
                    int64_t shown = 0;
                    for (int64_t o = beg; o < end && shown < max_nnz_to_print; ++o, ++shown) {
                        fprintf(stderr, " (%lld,%.2f)", (long long)col[o], val[o]);
                    }
                    if (nnz > max_nnz_to_print) fprintf(stderr, " ...");
                    fprintf(stderr, "\n");

                    // Validate first few entries in-row (col should be 0..)
                    int64_t lim = nnz;
                    if (lim > max_nnz_to_print) lim = max_nnz_to_print;
                    for (int64_t k = 0; k < lim; ++k) {
                        int64_t o = beg + k;
                        int64_t exp_col = k;
                        double exp_val = (double)(size - r);
                        if (col[o] != exp_col || fabs(val[o] - exp_val) > 1e-12) {
                            fprintf(stderr,
                                    "VAL/COL MISMATCH: rank=%d map=%d row=%lld k=%lld off=%lld got=(%lld,%.6g) exp=(%lld,%.6g)\n",
                                    rank, mapNo, (long long)r, (long long)k, (long long)o,
                                    (long long)col[o], val[o],
                                    (long long)exp_col, exp_val);
                            break;
                        }
                    }
                }
            }
        }
    }
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

    debug_check_migrated_slices("after rowD/valD/colD preserve",
                               rank,
                               size,
                               rowD,
                               row_var_p,
                               valD,
                               colD,
                               val_blk_p,
                               20,
                               8);


    // Verify local row_ptr slice correctness
    {
        int mapCount = laik_my_mapcount(row_var_p);
        for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
            int mrCount = laik_my_maprangecount(row_var_p, mapNo);
            assert(mrCount == 1);
            Laik_TaskRange* tr = laik_my_maprange(row_var_p, mapNo, 0);
            const Laik_Range* s = laik_taskrange_get_range(tr);
            int64_t rf = s->from.i[0];
            int64_t rt = s->to.i[0];

            int64_t* rp_base = NULL; uint64_t rp_len = 0;
            laik_get_map_1d(rowD, mapNo, (void**)&rp_base, &rp_len);

            int rm = -1;
            uint64_t lfrom_u = 0;
            Laik_Mapping* pm = laik_global2maplocal_1d(rowD, rf, &rm, &lfrom_u);
            assert(pm);
            assert(rm == mapNo);

            assert(lfrom_u < rp_len);
            assert((uint64_t)(rt - rf) <= (rp_len - lfrom_u));
            int64_t* rp = rp_base + (int64_t) lfrom_u;

            for (int64_t i = 0; i < (rt - rf); ++i) {
                int64_t g = rf + i;
                int64_t expected = g * (g - 1) / 2;
                if (rp[i] != expected) {
                    fprintf(stderr,"PREFIX MISMATCH rank=%d map=%d global_row=%lld rp=%lld exp=%lld\n",
                            rank, mapNo, (long long)g, (long long)rp[i], (long long)expected);
                    assert(0);
                }
            }
        }
    }

    // 7) Partition result
    Laik_Partitioning* res_blk_p = laik_new_partitioning(blk_pr, world, rows_space, 0);
    laik_switchto_partitioning(resD, res_blk_p, LAIK_DF_None, LAIK_RO_None);

    // 8) SpMV
    laik_set_phase(inst, 1, "SpMV", NULL);
    laik_iter_reset(inst);


    // debug prints
    {
        int mapCount = laik_my_mapcount(val_blk_p);
        for (int mapNo = 0; mapNo < mapCount; ++mapNo) {
            int mrCount = laik_my_maprangecount(val_blk_p, mapNo);
            assert(mrCount == 1);
            Laik_TaskRange* tr = laik_my_maprange(val_blk_p, mapNo, 0);
            const Laik_Range* s = laik_taskrange_get_range(tr);
            int64_t rowFrom = s->from.i[0];
            int64_t rowTo   = s->to.i[0];

            int rm = -1;
            uint64_t lfrom_u = 0;
            Laik_Mapping* pm = laik_global2maplocal_1d(rowD, rowFrom, &rm, &lfrom_u);
            assert(pm);
            const int64_t* rp_base = NULL; uint64_t rp_len = 0;
            laik_get_map_1d(rowD, rm, (void**)&rp_base, &rp_len);
            assert(lfrom_u < rp_len);
            assert((uint64_t)(rowTo - rowFrom + 1) <= (rp_len - lfrom_u));
            const int64_t* rp = rp_base + (int64_t) lfrom_u;

            double*  val = NULL; uint64_t val_len = 0;
            int64_t* col = NULL; uint64_t col_len = 0;
            laik_get_map_1d(valD, mapNo, (void**)&val, &val_len);
            laik_get_map_1d(colD, mapNo, (void**)&col, &col_len);
            assert(val_len == col_len);

            int64_t base = rp[0];
            int64_t maxRowsToPrint = 20;
            int64_t endRow = rowFrom + maxRowsToPrint;
            if (endRow > rowTo) endRow = rowTo;

            fprintf(stderr, "DEBUG rank=%d map=%d localRows=[%lld,%lld) rp_len=%llu val_len=%llu col_len=%llu\n",
                    rank, mapNo, (long long)rowFrom, (long long)rowTo,
                    (unsigned long long)rp_len,
                    (unsigned long long)val_len,
                    (unsigned long long)col_len);

            for (int64_t r = rowFrom; r < endRow; ++r) {
                int64_t li = r - rowFrom;
                int64_t offStart = rp[li]     - base;
                int64_t offEnd   = rp[li + 1] - base;
                int64_t nnz = offEnd - offStart;
                fprintf(stderr, "ROW %lld nnz=%lld :", (long long)r, (long long)nnz);
                int shown = 0;
                for (int64_t o = offStart; o < offEnd && shown < 5; ++o, ++shown) {
                    fprintf(stderr, " (%lld,%.2f)", (long long)col[o], val[o]);
                }
                if (nnz > 5) fprintf(stderr, " ...");
                fprintf(stderr, "\n");
            }
        }
    }

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
                int64_t nnz_row = end - beg;
                if (nnz_row != r) {
                    fprintf(stderr,"NNZ MISMATCH rank=%d map=%d row=%lld nnz=%lld exp=%lld\n",
                            rank, mapNo, (long long)r, (long long)nnz_row, (long long)r);
                    assert(0);
                }
            }

            for (int64_t r = fromRow; r < toRow; ++r) {
                int lr = (int)(r - fromRow);
                int64_t beg = row_ptr[lr]     - base;
                int64_t end = row_ptr[lr + 1] - base;
                for (int64_t o = beg; o < end; ++o)
                    res[lr] += val[o] * v[col[o]];
                laik_set_iteration(inst, (int)r);
            }

            for (int64_t r = fromRow; r < toRow; ++r) {
                int lr = (int)(r - fromRow);
                double expected = (double)(size - r) * 0.5 * (double)r * (double)(r + 1);
                if (fabs(res[lr] - expected) > 1e-9 && lr < 50) {
                    fprintf(stderr,"RES MISMATCH rank=%d map=%d row=%lld got=%g exp=%g\n",
                            rank, mapNo, (long long)r, res[lr], expected);
                    // assert(0);
                }
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