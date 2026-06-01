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
 * Naive SpMV example using the regular lexicographical layout.
 *
 * The sparse lower-triangular matrix is embedded into a dense MAXSIZE x MAXSIZE
 * 2D space, so each logical index stores exactly one value and repartitioning
 * moves dense row slabs instead of only the actual nonzeros.
 */

#include <laik.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// maximal size
#define MAXSIZE 10000

int main(int argc, char* argv[])
{
	Laik_Instance* inst = laik_init(&argc, &argv);
	Laik_Group* world = laik_world(inst);
	int rank = laik_myid(world);

	const char* prof_file = getenv("LAIK_PROFILE_FILE");
	int do_profile = (prof_file && prof_file[0] != '\0' && rank == 0);
	if (do_profile)
		laik_enable_profiling_file(inst, prof_file);

	int size = 0;
	if (argc > 1) size = atoi(argv[1]);
	if ((size <= 0) || (size > MAXSIZE)) size = MAXSIZE;

	laik_set_phase(inst, 0, "init", NULL);

	if (laik_myid(world) == 0) {
		const uint64_t dense_elems = (uint64_t) MAXSIZE * (uint64_t) MAXSIZE;
		const uint64_t nnz_per_row = 30;
		const uint64_t active_nnz = (uint64_t) size * nnz_per_row;
		printf("Naive dense lex layout: %llu stored values (%.2f MB), active nnz %llu\n",
			   (unsigned long long) dense_elems,
			   (double) (dense_elems * sizeof(double)) / (1024.0 * 1024.0),
			   (unsigned long long) active_nnz);
	}

	double* v = malloc(sizeof(double) * MAXSIZE);
	assert(v);
	for (int i = 0; i < MAXSIZE; i++)
		v[i] = (i < size) ? (double) (i + 1) : 0.0;

	// Dense matrix in a true 2D space, result on a 1D row space.
	Laik_Space* matrix_space = laik_new_space_2d(inst, MAXSIZE, MAXSIZE);
	Laik_Space* rows_space = laik_new_space_1d(inst, MAXSIZE);

	Laik_Data* matD = laik_new_data(matrix_space, laik_Double);
	Laik_Data* resD = laik_new_data(rows_space, laik_Double);

	// Initialize the dense matrix on the master process.
	Laik_Partitioner* master_pr = laik_new_master_partitioner();
	Laik_Partitioning* mat_master = laik_new_partitioning(master_pr, world, matrix_space, 0);
	laik_switchto_partitioning(matD, mat_master, LAIK_DF_None, LAIK_RO_None);

	if (laik_myid(world) == 0) {
		double* mat = NULL;
		uint64_t ysize = 0, ystride = 0, xsize = 0;
		laik_get_map_2d(matD, 0, (void**) &mat, &ysize, &ystride, &xsize);
		assert(ysize == (uint64_t) MAXSIZE);
		assert(xsize == (uint64_t) MAXSIZE);

		const uint64_t nnz_per_row = 30;
		for (uint64_t row = 0; row < ysize; row++) {
			uint64_t base = row * ystride;
			for (uint64_t col = 0; col < xsize; col++)
				mat[base + col] = 0.0;
			if (row < (uint64_t) size) {
				for (uint64_t k = 0; k < nnz_per_row; ++k) {
					uint64_t col = (row + k) % (uint64_t) size;
					mat[base + col] = 1.0;
				}
			}
		}
	}

	// Partition rows by the original work estimate.
	Laik_Partitioner* row_pr = laik_new_block_partitioner1();
	Laik_Partitioning* row_p = laik_new_partitioning(row_pr, world, rows_space, 0);
	laik_switchto_partitioning(resD, row_p, LAIK_DF_None, LAIK_RO_None);

	// Partition matrix rows directly (dimension 1), without copy partitioning.
	Laik_Partitioner* mat_pr = laik_new_block_partitioner(1, 1, 0, 0, 0);
	Laik_Partitioning* mat_p = laik_new_partitioning(mat_pr, world, matrix_space, 0);
	if (do_profile) {
		laik_profile_printf("# switch matD master->mat_p\n");
		laik_reset_profiling(inst);
	}
	laik_switchto_partitioning(matD, mat_p, LAIK_DF_Preserve, LAIK_RO_None);
	if (do_profile)
		laik_writeout_profile();

	const char* sleep_env = getenv("SLEEP_AFTER_INIT");
	if (sleep_env) {
		int seconds = atoi(sleep_env);
		if (seconds > 0)
			sleep((unsigned int)seconds);
	}

	// First SpMV: compute local rows and gather regularly.
	laik_set_phase(inst, 1, "1st SpmV", NULL);

	double* res = NULL;
	uint64_t count = 0;
	int64_t fromRow = 0, toRow = 0;
	laik_get_map_1d(resD, 0, (void**) &res, &count);
	for (uint64_t i = 0; i < count; i++)
		res[i] = 0.0;

	int64_t x1 = 0, x2 = 0, y1 = 0, y2 = 0;
	laik_my_range_1d(row_p, 0, &fromRow, &toRow);
	laik_my_range_2d(mat_p, 0, &x1, &x2, &y1, &y2);

	double* mat = NULL;
	uint64_t ysize = 0, ystride = 0, xsize = 0;
	laik_get_map_2d(matD, 0, (void**) &mat, &ysize, &ystride, &xsize);
	assert((uint64_t) (toRow - fromRow) == ysize);
	assert(y1 == fromRow && y2 == toRow);

	for (int64_t r = fromRow; r < toRow; r++) {
		uint64_t localRow = (uint64_t) (r - fromRow);
		for (uint64_t c = 0; c < xsize; c++)
			res[localRow] += mat[localRow * ystride + c] * v[c];
		laik_set_iteration(inst, (int) localRow);
	}

	Laik_Partitioning* pMaster = laik_new_partitioning(laik_Master, world, rows_space, 0);
	laik_switchto_partitioning(resD, pMaster, LAIK_DF_Preserve, LAIK_RO_None);
	if (laik_myid(world) == 0) {
		laik_get_map_1d(resD, 0, (void**) &res, &count);
		double sum = 0.0;
		for (uint64_t i = 0; i < count; i++) sum += res[i];
		printf("Res sum (regular): %f\n", sum);
	}

	// Second SpMV: use sum reduction to push the local results to the master.
	laik_iter_reset(inst);
	laik_set_phase(inst, 2, "2nd SpmV", NULL);

	Laik_Partitioning* pAll = laik_new_partitioning(laik_All, world, rows_space, 0);
	laik_switchto_partitioning(resD, pAll, LAIK_DF_Init, LAIK_RO_Sum);
	laik_get_map_1d(resD, 0, (void**) &res, &count);

	for (int64_t r = fromRow; r < toRow; r++) {
		uint64_t localRow = (uint64_t) (r - fromRow);
		for (uint64_t c = 0; c < xsize; c++)
			res[r] += mat[localRow * ystride + c] * v[c];
		laik_set_iteration(inst, (int) localRow);
	}

	laik_switchto_partitioning(resD, pMaster, LAIK_DF_Preserve, LAIK_RO_Sum);
	if (laik_myid(world) == 0) {
		laik_get_map_1d(resD, 0, (void**) &res, &count);
		double sum = 0.0;
		for (uint64_t i = 0; i < count; i++) sum += res[i];
		printf("Res sum (reduce): %f\n", sum);
	}

	free(v);
	laik_finalize(inst);
	return 0;
}
