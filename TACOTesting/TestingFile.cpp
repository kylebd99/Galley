#include <chrono>
#include<cstdlib> 
#include "taco/tensor.h"
#include "taco/format.h"
#include <iostream>

using namespace std::chrono;
using namespace taco;


void Experiment1(){
    std::cout << "Experiment # 1" << std::endl;
    const int DIM_SIZE = 100000000;
    Tensor<int> A("A", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> B("B", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> C("C", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> D("D", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> E("E", {DIM_SIZE}, Format({Sparse}));
    int numEntriesA = 10000000;
    int numEntriesB = 10000000;
    int numEntriesC = 1;
    
    for(int x = 0; x < numEntriesA; x++){
        A.insert({rand()%DIM_SIZE}, 12);
    }
    A.pack();
    for(int x = 0; x < numEntriesB; x++){
        B.insert({rand()%DIM_SIZE}, 12);
    }
    B.pack();
    for(int x = 0; x < numEntriesC; x++){
        C.insert({0}, 12);
    }
    C.pack();
    IndexVar i; 
    D(i) =  A(i) * B(i) * C(i);
    D.compile();
    D.assemble();
    auto start = high_resolution_clock::now();
    D.compute();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "D(i) Time: " << duration.count() << std::endl;

    E(i) = C(i) * A(i) * B(i) ;
    E.compile();
    E.assemble();
    start = high_resolution_clock::now();
    E.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "E(i) Time: " <<duration.count() << std::endl;
}

void Experiment2(){
    std::cout << "Experiment # 2" << std::endl;
    const int DIM_SIZE = 1000000;
    Tensor<int> A("A", {DIM_SIZE, DIM_SIZE}, Format({Sparse,Sparse}));
    Tensor<int> B("B", {DIM_SIZE, DIM_SIZE}, Format({Sparse,Sparse,}));
    Tensor<int> C("C", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> D("D", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> E("E", {DIM_SIZE}, Format({Sparse}));
    int numEntriesA = 10000000;
    int numEntriesB = 10000000;
    int numEntriesC = 1;
    
    for(int x = 0; x < numEntriesA; x++){
        A.insert({rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    A.pack();
    for(int x = 0; x < numEntriesB; x++){
        B.insert({rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    B.pack();
    for(int x = 0; x < numEntriesC; x++){
        C.insert({rand()%DIM_SIZE}, 12);
    }
    C.pack();
    IndexVar i, j, k; 
    D(i) =  A(i,j) * B(i,j) * C(j);
    D.compile();
    D.assemble();
    auto start = high_resolution_clock::now();
    D.compute();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "D(i) Time: " << duration.count() << std::endl;

    E(i) = C(j) * A(i,j) * B(i,j);
    E.compile();
    E.assemble();
    start = high_resolution_clock::now();
    E.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "E(i) Time: " <<duration.count() << std::endl;
}

void Experiment3(){
    std::cout << "Experiment # 3" << std::endl;
    const int DIM_SIZE = 10000;
    Tensor<int> A("A", {DIM_SIZE, DIM_SIZE, DIM_SIZE}, Format({Sparse, Sparse, Sparse}));
    Tensor<int> B("B", {DIM_SIZE, DIM_SIZE}, Format({Sparse,Sparse,}));
    Tensor<int> C("C", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> D("D", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> E("E", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> X("X", {DIM_SIZE,DIM_SIZE}, Format({Sparse,Sparse}));
    Tensor<int> F("F", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> Y("Y", {DIM_SIZE,DIM_SIZE}, Format({Sparse,Sparse}));
    Tensor<int> G("G", {DIM_SIZE}, Format({Sparse}));
    int numEntriesA = 10000000;
    int numEntriesB = 10000000;
    int numEntriesC = 1;
    
    for(int x = 0; x < numEntriesA; x++){
        A.insert({rand()%DIM_SIZE, rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    A.pack();
    for(int x = 0; x < numEntriesB; x++){
        B.insert({rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    B.pack();
    for(int x = 0; x < numEntriesC; x++){
        C.insert({rand()%DIM_SIZE}, 12);
    }
    C.pack();
    IndexVar i, j, k; 
    D(i) =  A(i,j,k) * B(i,j) * C(j);
    auto start = high_resolution_clock::now();
    D.compile();
    D.assemble();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "D(i) Compilation Time: " << duration.count() << std::endl;
    start = high_resolution_clock::now();
    D.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "D(i) Time: " << duration.count() << std::endl;

    E(i) =  A(k,i,j) * B(i,j) * C(j);
    start = high_resolution_clock::now();
    E.compile();
    E.assemble();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "F(i) Compilation Time: " << duration.count() << std::endl;
    start = high_resolution_clock::now();
    E.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "E(i) Time: " << duration.count() << std::endl;

    X(i, j) = A(k,i,j) * C(j);
    F(i) = X(i,j) * B(i,j);
    start = high_resolution_clock::now();
    F.compile();
    F.assemble();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "F(i) Compilation Time: " << duration.count() << std::endl;
    start = high_resolution_clock::now();
    F.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "F(i) Time: " << duration.count() << std::endl;

    Y(i, j) = A(k,i,j) * B(i,j);
    G(i) = Y(i,j) * C(j);
    start = high_resolution_clock::now();
    G.compile();
    G.assemble();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "G(i) Compilation Time: " << duration.count() << std::endl;
    start = high_resolution_clock::now();
    G.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "G(i) Time: " << duration.count() << std::endl;
}


void Experiment4(){
    std::cout << "Experiment # 4" << std::endl;
    const int DIM_SIZE = 100000;
    Tensor<int> A("A", {DIM_SIZE, DIM_SIZE, DIM_SIZE}, Format({Sparse, Sparse, Sparse}));
    Tensor<int> B("B", {DIM_SIZE, DIM_SIZE, DIM_SIZE}, Format({Sparse, Sparse, Sparse,}));
    Tensor<int> C("C", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> D("D", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> E("E", {DIM_SIZE}, Format({Sparse}));
    int numEntriesA = 100000000;
    int numEntriesB = 100000000;
    int numEntriesC = 1;
    
    for(int x = 0; x < numEntriesA; x++){
        A.insert({rand()%DIM_SIZE, rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    A.pack();
    for(int x = 0; x < numEntriesB; x++){
        B.insert({rand()%DIM_SIZE, rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    B.pack();
    for(int x = 0; x < numEntriesC; x++){
        C.insert({1}, 12);
    }
    C.pack();
    IndexVar i("i"), j("j"), k("k"); 
    D(i) =  A(i, j, k) * B(i, j, k) * C(j);
    IndexStmt stmt = D.getAssignment().concretize();
    stmt.reorder({k, i, j});
    auto start = high_resolution_clock::now();
    D.compile(stmt);
    D.assemble();
    D.compute();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "D(i) Time: " << duration.count() << std::endl;

    Access matrix =  C(j);
    E(i) =  A(i, j, k) * B(i,j, k) * matrix;
    stmt = E.getAssignment().concretize();
    stmt.reorder({j,i,k});
    start = high_resolution_clock::now();
    E.compile(stmt);
    E.assemble();
    E.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "E(i) Time: " << duration.count() << std::endl;
}

void Experiment5(){
    std::cout << "Experiment # 5" << std::endl;
    const int DIM_SIZE = 1000;
    Tensor<int> A("A", {DIM_SIZE, DIM_SIZE, DIM_SIZE}, Format({Sparse, Sparse, Sparse}));
    Tensor<int> B("B", {DIM_SIZE, DIM_SIZE, DIM_SIZE}, Format({Sparse, Sparse, Sparse,}));
    Tensor<int> C("C", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> D("D", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> E("E", {DIM_SIZE}, Format({Sparse}));
    Tensor<int> F("F", {DIM_SIZE}, Format({Sparse}));
    int numEntriesA = 10000000;
    int numEntriesB = 10000000;
    int numEntriesC = 1000;
    
    for(int x = 0; x < numEntriesA; x++){
        A.insert({rand()%DIM_SIZE, rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    A.pack();
    for(int x = 0; x < numEntriesB; x++){
        B.insert({rand()%DIM_SIZE, rand()%DIM_SIZE, rand()%DIM_SIZE}, 12);
    }
    B.pack();
    for(int x = 0; x < numEntriesC; x++){
        C.insert({rand()%DIM_SIZE}, 12);
    }
    C.pack();
    IndexVar i("i"), j("j"), k("k"); 
    D(i) =  A(i, j, k) * B(i, j, k) * C(j);
    IndexStmt stmt = D.getAssignment().concretize();
    stmt.reorder({k, i, j});
    auto start = high_resolution_clock::now();
    D.compile(stmt);
    D.assemble();
    D.compute();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "D(i) Time: " << duration.count() << std::endl;

    IndexVar k1, k2;
    F(i) =  A(i, j, k) * B(i,j, k) * C(j);
    stmt = F.getAssignment().concretize();
    stmt.reorder({k, i, j});
    stmt.split(k, k1, k2, 10);
    stmt.parallelize(k1, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);z
    start = high_resolution_clock::now();
    F.compile(stmt);
    F.assemble();
    F.compute();
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    std::cout << "F(i) Time: " << duration.count() << std::endl;
}


int main(int numArgs, char** args){
//    Experiment1();
//    Experiment2();
//    Experiment3();
//    Experiment4();
    Experiment5();
    return 0;
}
