#include <apf_planner/functions.h>
#include <iostream>
#include <cmath>    //inclusione libreria matematica per eseguire radici quadrate
#include <cstdlib>

#define DIMENSION 2 //dimensione del problema (due dimensioni, x e y)


//***********FUNZIONI UTILI PER ESEGUIRE VARIE TASK PRELIMINARI PER IL CALCOLO DELLE FORZE (direttamente all'interno del file apf.cppp):*************

//CALCOLO DISTANZA (norma del vettore differenza tra due vettori, vec1 e vec2)
double vect_norm2(std::vector<double> vec1, std::vector<double> vec2){
        double norma;
        std::vector<double> difference={0,0};
        
        for (int i=0; i<DIMENSION; i++){
            difference[i]=vec1[i]-vec2[i];
        }
        
        norma = sqrt(difference[0]*difference[0]+difference[1]*difference[1]);
        return norma;
    }

//CALCOLO NORMA DI UN VETTORE
double vect_norm1(std::vector<double> vec1){
        double norma=0;
        norma = sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]);
        return norma;
    }

//STABILISCE IL VERSORE DIREZIONE RIVOLTO DA vec1 A vec2
std::vector<double> compute_direction(std::vector<double> vec1, std::vector<double> vec2){
        
        std::vector<double> dir={0,0};
        double norm = vect_norm2(vec1, vec2);

        if (norm<0.01) norm=0.01; //se tende a zero allora la forza diventa troppo grande e può causare problemi

        for(int i=0; i<DIMENSION; i++){
            dir[i]=(vec1[i]-vec2[i])/norm;
        }

        return dir;
    }
    
//CALCOLA IL COSENO DELL'ANGOLO TRA I DUE VETTORI
double compute_cos_gamma(std::vector<double> vec1, std::vector<double> vec2){
    double coseno=0;
    for(int i=0; i<vec1.size(); i++){
        vec2[i]=-vec2[i];
    }

    coseno=(vec1[0]*vec2[0]+vec1[1]*vec2[1])/(vect_norm1(vec1)*vect_norm1(vec2));
    return coseno;
}

//FUNZIONE SEGNO:
int sign(double x){
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
