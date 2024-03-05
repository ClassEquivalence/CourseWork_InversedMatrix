#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//если значение меньше этой величины - считаем его нулевым.
constexpr double zero_error = 0.00000001;
//зерно генерации псевдослучайных чисел srand(100).
constexpr int matrix_gen_seed = 100;
//искуственно ограничиваем разброс генерируемых значений.
constexpr int randlim = 32768;
//-1 - матрица вырождена. -2 - матрица невырождена(вырожденность ещё не доказана).
constexpr int ok_singular = -1, not_singular = -2;

//код ошибки связанный с неверной размерностью матрицы
constexpr int ERROR_EXIT_CODE_DIMEXCEEDED = 10;
//код ошибки связанный с вырожденностью матрицы.
constexpr int ERROR_EXIT_CODE_SINGULAR = -123;

//ЗДЕСЬ ЗАДАЁТСЯ РАЗМЕРНОСТЬ МАТРИЦЫ
int mdim;

//разряд переменной int занят сообщением о том,
//что мы выделили для матрицы память на процессоре
constexpr int flag_initCPU = 1;

//в эту переменную записываем максимальное число потоков,
//поддерживаемых видеокартой в рамках 1 блока
int maxdevicethreads;
//указатель на память видеокарты с информацией о вырожденности матрицы,
//и ещё кое-какой информацией
int* issingular;

struct Matrix {
	//указатель на область памяти в процессоре и на видеокарте
	double* cpu_mptr, * dev_mptr;
	int dim;
	int flags; // 1 - есть ли CPUMem
};

struct Mptrs {
	Matrix* mptr, * imptr;
};

//инициализируем начальные данные, выделяем память, необходимую для корректной
//работы программы. Матрицы пока не инициализируем
void initGJData();
//Теперь инициализируем данные структуры матрица, выделяем память под неё.
Matrix initMatrix(int dim, bool initCPUMemToo);
//получаем обратную матрицу к поданной на вход
Matrix inversed(Matrix* cpu_mptr, Matrix* dev_mptr);
//копируем матрицу(с новым выделением памяти)
Matrix copyMatrix(Matrix* mptr);
//удаляем матрицу
void deleteMatrix(Matrix* mptr);
//генерируем случайные значения в матрице
void initMatrixRandomValues(Matrix* mptr);
//генерируем значения единичной матрицы
void initUnitMatrixValues(Matrix* mptr, int dim);
//инициализируем матрицу с консольного ввода
Matrix initMatrixFromInput();
//инициализируем матрицу из массива, объявленного в коде.
Matrix initMatrixFromCode(double* arr, int dim);
//выделяем память на GPU под структуру(и только под структуру. память уже выделяли)
void cudaManageMptrs(Matrix** mptr, Matrix* cpu_mptr);
//Функция вывода значений матрицы. Полезна для тестирования.
void pmfunc(Matrix* m, bool isinversed);
//функция вычисляет коэффициенты для функции ниже.
__global__
void strmult_quotient_calculations(double* q, Matrix* mptr);
//функция умножения строк матрицы на коэффициенты массива q
__global__
void str_multiply(double* q, Matrix* mptr);
//функция создаёт на памяти GPU значения единичной матрицы
__global__
void make_unit_matrix(Matrix mptr);
//функция исполняется на GPU одним блоком и потоком, выводя значения матрицы.
//служит для функции pmfunc
__global__
void dev_print_matrix(Matrix* mptr);
//функция вычисляет коэффициенты для функции ниже.
__global__
void strsum_quotient_calculations(double* q, int s, Matrix* mptr);
//функция прибавляет к строкам матрицы строку под номером s, умноженную на
//нужные коэффициенты массива q
__global__
void dev_str_sum(double* q, int s, Matrix* mptr);
//функция проверяет, можно ли на текущем шаге s доказать вырожденность матрицы.
// если можно - завершает работу программы с выводом сообщения о вырожденности.
// если она ни разу вырожденность не доказала за время работы - матрица точно
// невырожденна, в теории...
//а ещё функция делает главную диагональ ненулевой, прибавляя строку с помощью
//функции ниже, если это необходимо.
bool not_zero_diag(Mptrs mptrs, int dim, int s, double* q);
//функция прибавляет к одной строке другую умноженную на k
__global__
void dev_str_sum_bystrnums(Matrix* mptr, int s1, int s2, double k);
//эта функция служит функции not_zero_diag, проверяя, есть ли на s-ом месте
// главной диагонали ненулевое значение, и если его можно сделать ненулевым добавив
// строку - то записывает строку в q, а если нельзя - то сообщает о вырожденности
//опять же о вырожденности докладывает через q.
__global__
void restore_obviousity_singularity(Matrix* mptr, int s, int* q);
//Функция умножения матриц. Позволяет убедиться, что обратная матрица найдена верно.
//AA^-1=E.
Matrix matrixMult(Matrix* m1, Matrix* m2);
//функция подчинена функции выше.
__global__
void dev_MatrixMult(double* mres, double* m1, double* m2, int dim);



int main() {
	FILE *fp;
	char fname[] = "Matrixinput.txt";
	if ((fp = fopen(fname, "r")) == NULL)
	{
		printf("Не удалось открыть файл");
		return 0;
	}
	fscanf(fp, "%d", &mdim);
	double* minp;
	minp = (double*)malloc(sizeof(double) * mdim * mdim);
	for (int i = 0; i < mdim; i++) {
		for (int j = 0; j < mdim; j++) {
			fscanf(fp, "%lf ", &minp[i * mdim + j]);
		}
	}

	cudaDeviceGetAttribute(&maxdevicethreads,
		cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock, 0);
	initGJData();


	Matrix m_fromcode = initMatrixFromCode(minp, mdim), * dev_m_fromcode;
	cudaManageMptrs(&dev_m_fromcode, &m_fromcode);
	pmfunc(dev_m_fromcode, false);

	/*
	Matrix* dev_mptr;
	Matrix cpu_mptr;
	cpu_mptr = initMatrix(mdim, true);
	initMatrixRandomValues(&cpu_mptr);
	cudaManageMptrs(&dev_mptr, &cpu_mptr);
	cudaDeviceSynchronize();
	pmfunc(dev_mptr, false);
	*/

	Matrix inversedM = inversed(&m_fromcode, dev_m_fromcode);
	Matrix* dev_inversedM;
	cudaManageMptrs(&dev_inversedM, &inversedM);
	pmfunc(dev_inversedM, true);

	Matrix multres = matrixMult(&m_fromcode, &inversedM);
	Matrix* dev_multres;
	cudaManageMptrs(&dev_multres, &multres);
	pmfunc(dev_multres, false);

	//здесь можно было бы прописать cudaFree и т.п...
}

Matrix inversed(Matrix* cpu_mptr, Matrix* dev_mptr) {
	if (cpu_mptr->dim > maxdevicethreads)
	{
		printf("Максимально допустимая размерность: %d", maxdevicethreads);
		printf("Максимально допустимая размерность при этом превышена(%d).\n"
			"Программа завершает работу.", cpu_mptr->dim);
		exit(ERROR_EXIT_CODE_DIMEXCEEDED);
	}
	else if (cpu_mptr->dim <= 0) {
		printf("Размерность должна быть положительной");
		exit(ERROR_EXIT_CODE_DIMEXCEEDED);
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double* quotients;
	cudaMalloc(&quotients, sizeof(double) * cpu_mptr->dim);
	Matrix copied = copyMatrix(cpu_mptr), * dev_copied;
	Matrix cpu_imptr, * dev_imptr;
	cpu_imptr = initMatrix(copied.dim, true);
	initUnitMatrixValues(&cpu_imptr, copied.dim);
	cudaManageMptrs(&dev_imptr, &cpu_imptr);
	cudaManageMptrs(&dev_copied, &copied);
	Mptrs dev_mptrs = { dev_copied, dev_imptr };
	bool isnt_zero;
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	for (int s = 0; s < copied.dim; s++) {
		isnt_zero = not_zero_diag(dev_mptrs, copied.dim, s, quotients);
		if (!isnt_zero)
		{
			printf("Матрица вырождена.");
			exit(ERROR_EXIT_CODE_SINGULAR);
		}
		strsum_quotient_calculations << <1, copied.dim >> > (quotients, s, dev_copied);

		dev_str_sum << <copied.dim, copied.dim >> > (quotients, s, dev_copied);
		dev_str_sum << <copied.dim, copied.dim >> > (quotients, s, dev_imptr);
	}


	strmult_quotient_calculations << <1, copied.dim >> > (quotients, dev_copied);
	//str_multiply << <copied.dim, copied.dim >> > (quotients, dev_copied);
	str_multiply << <copied.dim, copied.dim >> > (quotients, dev_imptr);

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Time spent(ms): %f\n", time);

	deleteMatrix(&copied);
	cudaFree(dev_copied);
	cudaFree(dev_imptr);
	cudaFree(quotients);

	return cpu_imptr;
}

void initGJData() {
	cudaMalloc(&issingular, sizeof(int));
}

Matrix initMatrix(int dim, bool initCPUmemToo) {
	Matrix tobeInit;
	tobeInit.dim = dim;
	if (initCPUmemToo)
		tobeInit.cpu_mptr = (double*)malloc(sizeof(double) * dim * dim);
	cudaMalloc(&tobeInit.dev_mptr, sizeof(double) * dim * dim);
	cudaDeviceSynchronize();
	tobeInit.flags = 0;
	if (initCPUmemToo)
		tobeInit.flags = tobeInit.flags | flag_initCPU;
	return tobeInit;
}

Matrix copyMatrix(Matrix* mptr) {
	Matrix tobeInit;
	int dim = tobeInit.dim = mptr->dim;
	tobeInit.flags = mptr->flags;
	if (tobeInit.flags & flag_initCPU) {
		tobeInit.cpu_mptr = (double*)malloc(sizeof(double) * dim * dim);
		cudaMemcpy(tobeInit.cpu_mptr, mptr->cpu_mptr, sizeof(double) * dim * dim,
			cudaMemcpyHostToHost);
	}
	cudaMalloc(&tobeInit.dev_mptr, sizeof(double) * dim * dim);
	cudaMemcpy(tobeInit.dev_mptr, mptr->dev_mptr, sizeof(double) * dim * dim,
		cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	return tobeInit;
}

void deleteMatrix(Matrix* mptr) {
	if ((mptr->flags | flag_initCPU) == mptr->flags)
		free(mptr->cpu_mptr);
	cudaFree(mptr->dev_mptr);
}
void initMatrixRandomValues(Matrix* mptr) {
	srand(matrix_gen_seed);
	for (int i = 0; i < mptr->dim; i++) {
		for (int j = 0; j < mptr->dim; j++) {
			mptr->cpu_mptr[i * mptr->dim + j] = rand() % 6;
		}
	}
	cudaMemcpy(mptr->dev_mptr, mptr->cpu_mptr, sizeof(double) *
		mptr->dim * mptr->dim, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}


//инициализирует только gpu-часть.
void initUnitMatrixValues(Matrix* mptr, int dim) {
	make_unit_matrix << <dim, dim >> > (*mptr);
}

Matrix initMatrixFromInput() {
	int dim;
	printf("Введите размерность матрицы:");
	scanf("%d", &dim);
	Matrix tobeInit = initMatrix(dim, true);
	printf("\nА теперь введите сами значения матрицы:\n");
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			scanf("%lf ", &tobeInit.cpu_mptr[i * dim + j]);
		}
	}
	cudaMemcpy(tobeInit.dev_mptr, tobeInit.cpu_mptr, sizeof(double) * dim * dim,
		cudaMemcpyHostToDevice);
	return tobeInit;
}


Matrix initMatrixFromCode(double* arr, int dim) {
	Matrix tobeInit = initMatrix(dim, true);
	cudaMemcpy(tobeInit.cpu_mptr, arr, sizeof(double) * dim * dim,
		cudaMemcpyHostToHost);
	cudaMemcpy(tobeInit.dev_mptr, arr, sizeof(double) * dim * dim,
		cudaMemcpyHostToDevice);
	return tobeInit;
}

void cudaManageMptrs(Matrix** dev_mptr, Matrix* cpu_mptr) {
	cudaMalloc(dev_mptr, sizeof(Matrix));
	cudaMemcpy(*dev_mptr, cpu_mptr, sizeof(Matrix), cudaMemcpyHostToDevice);
}


__global__
void make_unit_matrix(Matrix mptr) {
	int index2d = blockIdx.x * mptr.dim + threadIdx.x;
	if (blockIdx.x != threadIdx.x)
		mptr.dev_mptr[index2d] = 0;
	else
		mptr.dev_mptr[index2d] = 1;
}

__global__
void strsum_quotient_calculations(double* q, int s, Matrix* mptr) {
	if (s == threadIdx.x)
		q[threadIdx.x] = 0;
	else
		q[threadIdx.x] = mptr->dev_mptr[threadIdx.x * mptr->dim + s] /
		mptr->dev_mptr[s * mptr->dim + s];
}

__global__
void dev_str_sum(double* q, int s, Matrix* mptr) {
	mptr->dev_mptr[blockIdx.x * mptr->dim + threadIdx.x] -=
		mptr->dev_mptr[s * mptr->dim + threadIdx.x] * q[blockIdx.x];
}

__global__
void strmult_quotient_calculations(double* q, Matrix* mptr) {
	q[threadIdx.x] = 1 / mptr->dev_mptr[threadIdx.x * mptr->dim + threadIdx.x];
}

//умножаем строки, довершая работу.
__global__
void str_multiply(double* q, Matrix* mptr) {
	mptr->dev_mptr[blockIdx.x * mptr->dim + threadIdx.x] *= q[blockIdx.x];
}

__global__
void dev_print_matrix(Matrix* mptr) {
	for (int i = 0; i < mptr->dim; i++) {
		for (int j = 0; j < mptr->dim; j++) {
			printf("%lf ", mptr->dev_mptr[i * mptr->dim + j]);
		}
		printf("\n");
	}
}


bool not_zero_diag(Mptrs mptrs, int dim, int s, double* q) {
	restore_obviousity_singularity << <1, 1 >> > (mptrs.mptr, s, issingular);
	int result;
	cudaMemcpy(&result, issingular, sizeof(int), cudaMemcpyDeviceToHost);
	if (result == ok_singular)
		return false;
	else if (result == not_singular)
		return true;
	else
	{
		dev_str_sum_bystrnums << <1, dim >> > (mptrs.mptr, s, result, 1);
		dev_str_sum_bystrnums << <1, dim >> > (mptrs.imptr, s, result, 1);
		return true;
	}
}

__global__
void dev_str_sum_bystrnums(Matrix* mptr, int s1, int s2, double k) {
	mptr->dev_mptr[s1 * mptr->dim + threadIdx.x] += mptr->dev_mptr[s2 * mptr->dim + threadIdx.x] * k;
}

//Если всё норм и ничё не надо менять то -1.
//Если матрица неединичная, то -2.
//Если надо добавить строку - вернуть число-номер строки.
__global__
void restore_obviousity_singularity(Matrix* mptr, int s, int* q) {
	if (fabs(mptr->dev_mptr[s * mptr->dim + s]) > zero_error)
	{
		*q = not_singular; //не вырождена
		return;
	}
	else
	{
		for (int i = s + 1; i < mptr->dim; i++) {
			if (fabs(mptr->dev_mptr[i * mptr->dim + s]) > zero_error)
			{
				*q = i;
				return;
			}
		}
	}
	*q = ok_singular; //вырождена
}

void pmfunc(Matrix* dev_m, bool isinversed) {
	if (!isinversed)
		printf("Матрица:\n");
	else
		printf("Обратная матрица:\n");

	dev_print_matrix << <1, 1 >> > (dev_m);
	cudaDeviceSynchronize();
}

Matrix matrixMult(Matrix* m1, Matrix* m2) {
	Matrix resmatrix = initMatrix(m1->dim, true);

	dev_MatrixMult << <m1->dim, m1->dim >> > (resmatrix.dev_mptr, m1->dev_mptr,
		m2->dev_mptr, m1->dim);

	cudaMemcpy(resmatrix.cpu_mptr, resmatrix.dev_mptr,
		sizeof(double) * m1->dim * m1->dim, cudaMemcpyDeviceToHost);

	return resmatrix;
}

__global__
void dev_MatrixMult(double* mres, double* m1, double* m2, int dim) {
	double sum = 0;
	for (int i = 0; i < dim; i++) {
		sum += m1[blockIdx.x * dim + i] * m2[i * dim + threadIdx.x];
	}
	mres[blockIdx.x * dim + threadIdx.x] = sum;
}
