#include <iostream>
#include "unsupported/Eigen/CXX11/Tensor"
#include <Fastor/Fastor.h>

constexpr int FIFTY =  50;
constexpr int TWETY =  20;
constexpr int TEN =  10;
constexpr int FOUR =  4;

using real_ = double;

real_ A [TWETY][FIFTY][TEN]  = {1} ;
real_ B [FIFTY][TEN][TWETY][FOUR]  = {1};
real_ C [FOUR]  = {};
real_ D [TEN][TEN][FOUR]  = {};

Fastor::Tensor<real_,TWETY,FIFTY,TEN> fA;
Fastor::Tensor<real_,FIFTY,TEN,TWETY,FOUR> fB;
Fastor::Tensor<real_,FOUR> fC_loop;
Fastor::Tensor<real_,TEN,TEN,FOUR> fD_loop;

Fastor::Tensor<real_,TWETY,FIFTY,TEN> fA_col;
Fastor::Tensor<real_,FIFTY,TEN,TWETY,FOUR> fB_col;
Fastor::Tensor<real_,FOUR> fC_loop_col;
Fastor::Tensor<real_,TEN,TEN,FOUR> fD_loop_col;

Eigen::TensorFixedSize<real_, Eigen::Sizes<TWETY, FIFTY, TEN>> eA;
Eigen::TensorFixedSize<real_, Eigen::Sizes<FIFTY,TEN, TWETY, FOUR>> eB;
Eigen::TensorFixedSize<real_, Eigen::Sizes<FOUR>> eC;
Eigen::TensorFixedSize<real_, Eigen::Sizes<TEN,TEN,FOUR>> eD;
Eigen::TensorFixedSize<real_, Eigen::Sizes<FOUR>> eC_loop;
Eigen::TensorFixedSize<real_, Eigen::Sizes<TEN,TEN,FOUR>> eD_loop;


void CTran_C() {

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                A[i][j][k] = 1.0;

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int l = 0; l < FOUR; l++)
                    B[j][k][i][l] = 1.0;

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int l = 0; l < FOUR; l++)
                    C[l] += A[i][j][k] * B[j][k][i][l];

    Fastor::unused(A);
    Fastor::unused(B);
    Fastor::unused(C);
}

void CTran_D() {

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                A[i][j][k] = 1.0;

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int l = 0; l < FOUR; l++)
                    B[j][k][i][l] = 1.0;

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int m = 0; m < TEN; m++)
                    for ( int n = 0; n < FOUR; n++)
                        D[k][m][n] += A[i][j][k] * B [j][m][i][n];

    Fastor::unused(A);
    Fastor::unused(B);
    Fastor::unused(D);
}


void Fastor_C() {

    using namespace Fastor;
    enum {I,J,K,L,M,N};

    fA.ones();
    fB.ones();

    auto fC = einsum<Index<I,J,K>,Index<J,K,I,M>>(fA,fB);

    unused(fA);
    unused(fB);
    unused(fC);
}

void Fastor_D() {

    using namespace Fastor;
    enum {I,J,K,L,M,N};

    fA.ones();
    fB.ones();

    auto fD = einsum<Index<I,J,K>,Index<J,M,I,N>>(fA,fB);

    unused(fA);
    unused(fB);
    unused(fD);
}


void Eigen_C() {

    using namespace Eigen;

    eA.setConstant(1);
    eB.setConstant(1);

    array<IndexPair<int>,3> IJK_JKIM = {
        IndexPair<int>(0, 2),
        IndexPair<int>(1, 0),
        IndexPair<int>(2, 1),
    };

    eC = eA.contract(  eB,  IJK_JKIM) ;

    Fastor::unused(eA);
    Fastor::unused(eB);
    Fastor::unused(eC);
}


void Eigen_D() {
    using namespace Eigen;

    eA.setConstant(1);
    eB.setConstant(1);

     array<IndexPair<int>,2> IJK_JMIN = {
         IndexPair<int>(0, 2),
         IndexPair<int>(1, 0),
     };

    eD = eA.contract(  eB,  IJK_JMIN) ;

    Fastor::unused(eA);
    Fastor::unused(eB);
    Fastor::unused(eD);
}


void Fastor_C_loop() {

    using namespace Fastor;
    enum {I,J,K,L,M,N};

    fA.ones();
    fB.ones();

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int l = 0; l < FOUR; l++)
                    fC_loop(l) += fA(i, j, k) * fB(j, k, i, l);

    unused(fA);
    unused(fB);
    unused(fC_loop);
}

void Fastor_D_loop() {

    using namespace Fastor;
    enum {I,J,K,L,M,N};

    fA.ones();
    fB.ones();

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int m = 0; m < TEN; m++)
                    for ( int n = 0; n < FOUR; n++)
                        fD_loop(k, m, n) += fA(i, j, k) * fB(j, m, i, n);

    unused(fA);
    unused(fB);
    unused(fD_loop);
}


void Eigen_C_loop() {

    using namespace Eigen;

    eA.setConstant(1);
    eB.setConstant(1);

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int l = 0; l < FOUR; l++)
                    eC_loop(l) += eA(i, j, k) * eB(j, k, i, l);

    Fastor::unused(eA);
    Fastor::unused(eB);
    Fastor::unused(eC_loop);
}


void Eigen_D_loop() {
    using namespace Eigen;

    eA.setConstant(1);
    eB.setConstant(1);

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int m = 0; m < TEN; m++)
                    for ( int n = 0; n < FOUR; n++)
                        eD_loop(k, m, n) += eA(i, j, k) * eB(j, m, i, n);

    Fastor::unused(eA);
    Fastor::unused(eB);
    Fastor::unused(eD_loop);
}


Eigen::TensorFixedSize<real_, Eigen::Sizes<TWETY, FIFTY, TEN>> e_loop_a;
Eigen::TensorFixedSize<real_, Eigen::Sizes<TWETY, FIFTY, TEN, FOUR>> e_loop_b;
Eigen::TensorFixedSize<real_, Eigen::Sizes<TWETY, FIFTY, TEN>> e_loop_c;

Fastor::Tensor<real_,TWETY, FIFTY, TEN> f_loop_a;
Fastor::Tensor<real_,TWETY, FIFTY, TEN, FOUR> f_loop_b;
Fastor::Tensor<real_,TWETY, FIFTY, TEN> f_loop_c;

void eigen_loop() {
    using namespace Eigen;

    e_loop_a.setConstant(1);
    e_loop_b.setConstant(1);

    for ( int n = 0; n < FOUR; n++)
        for ( int k = 0; k < TEN; k++)
            for ( int j = 0; j < FIFTY; j++)
                for ( int i = 0; i < TWETY; i++)
                    e_loop_c(i, j, k) += e_loop_a(i, j, k) * e_loop_b(i, j, k, n);

    Fastor::unused(e_loop_a);
    Fastor::unused(e_loop_b);
    Fastor::unused(e_loop_c);
}


void eigen_reverse_loop() {
    using namespace Eigen;

    eA.setConstant(1);
    eB.setConstant(1);

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int n = 0; n < FOUR; n++)
                    e_loop_c(i, j, k) += e_loop_a(i, j, k) * e_loop_b(i, j, k, n);

    Fastor::unused(e_loop_a);
    Fastor::unused(e_loop_b);
    Fastor::unused(e_loop_c);
}


void fastor_loop() {
    using namespace Fastor;
    enum {I,J,K,L,M,N};

    f_loop_a.ones();
    f_loop_b.ones();

    for ( int n = 0; n < FOUR; n++)
        for ( int k = 0; k < TEN; k++)
            for ( int j = 0; j < FIFTY; j++)
                for ( int i = 0; i < TWETY; i++)
                    f_loop_c(i, j, k) += f_loop_a(i, j, k) * f_loop_b(i, j, k, n);

    unused(f_loop_a);
    unused(f_loop_b);
    unused(f_loop_c);
}


void fastor_reverse_loop() {
    using namespace Fastor;
    enum {I,J,K,L,M,N};

    f_loop_a.ones();
    f_loop_b.ones();

    for ( int i = 0; i < TWETY; i++)
        for ( int j = 0; j < FIFTY; j++)
            for ( int k = 0; k < TEN; k++)
                for ( int n = 0; n < FOUR; n++)
                    f_loop_c(i, j, k) += f_loop_a(i, j, k) * f_loop_b(i, j, k, n);

    unused(f_loop_a);
    unused(f_loop_b);
    unused(f_loop_c);
}

void Fastor_C_col_major() {

    using namespace Fastor;
    enum {I,J,K,L,M,N};

    fA_col.ones();
    fB_col.ones();

    auto fC_col = einsum<Index<I,J,K>,Index<J,K,I,M>>(fA_col,fB_col);

    unused(fA_col);
    unused(fB_col);
    unused(fC_col);
}

void Fastor_D_col_major() {

    using namespace Fastor;
    enum {I,J,K,L,M,N};

    fA_col.ones();
    fB_col.ones();

    auto fD_col = einsum<Index<I,J,K>,Index<J,M,I,N>>(fA_col,fB_col);

    unused(fA_col);
    unused(fB_col);
    unused(fD_col);
}


int main() {
    using namespace Fastor;

    print("Time for computing tensor C (CTran, Fastor, Eigen, Fastor Loop, Eigen loop):");
    timeit(CTran_C);
    timeit(Fastor_C);
    timeit(Eigen_C);
    timeit(Fastor_C_loop);
    timeit(Eigen_C_loop);
    print("Time for computing tensor D (CTran, Fastor, Eigen, Fastor loop, Eigen loop):");
    timeit(CTran_D);
    timeit(Fastor_D);
    timeit(Eigen_D);
    timeit(Fastor_D_loop);
    timeit(Eigen_D_loop);
    print("Timing looping (Eigen correct order, Eigen reverse, Fastor correct order, Fastor reverse)");
    timeit(eigen_loop);
    timeit(eigen_reverse_loop);
    timeit(fastor_loop);
    timeit(fastor_reverse_loop);
    print("Timing fastor einstein notation with (C row major, C col major, D row major, D col major)");
    timeit(Fastor_C);
    timeit(Fastor_C_col_major);
    timeit(Fastor_D);
    timeit(Fastor_D_col_major);


    return 0;
}