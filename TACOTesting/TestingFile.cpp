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
    const int DIM_SIZE = 1000000;
    Tensor<int> A("A", {DIM_SIZE, DIM_SIZE, DIM_SIZE}, Format({Sparse, Sparse, Sparse}));
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

int main(int numArgs, char** args){
//    Experiment1();
    Experiment2();
    return 0;
}
