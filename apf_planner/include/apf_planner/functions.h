#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include <vector>

//DEFINIZIONE ALTRE FUNZIONI UTILI:
double vect_norm2(std::vector<double> vec1, std::vector<double> vec2);

double vect_norm1(std::vector<double> vec1);

std::vector<double> compute_direction(std::vector<double> vec1, std::vector<double> vec2);

double compute_cos_gamma(std::vector<double> vec1, std::vector<double> vec2); //calcolo del coseno gamma per poter determinare successivamente l'effetto della forza repulsiva

int sign(double expression);

#endif