#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <numeric>

using namespace std;

template <typename T>
vector<T> range(T start, T end, T step = 1)
{
    /*
    ** Creates a vector of numbers in the form [start, end-1],
    ** separated by <step>.
    **
    ** @param start - Starting value for the sequence, inclusive.
    ** @param end - Stop value, not included in the sequence.
    ** @return - Vector containing the sequence
    */
    int length = abs(end - start) / step;
    vector<T> result;
    for (int i = 0; i <= length; i++)
        result.push_back(start + i * step);
    return result;
}

template <typename T>
ostream &operator<<(ostream &os, const vector<T> v)
{
    // Overloads "<<" operator to print all elements in vector <v>
    int size = v.size();
    for (int i = 0; i < size - 1; i++)
        os << v[i] << ' ';
    os << v[size - 1];
    return os;
}

template <typename T>
ostream &operator<<(ostream &os, const vector<vector<T>> v)
{
    // Overloads "<<" operator to print all elements matrix <v>
    int rows = v.size();
    int cols = v[0].size();

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            os << setw(2) << v[i][j] << setw(2) << ' ';
        }
        os << setw(2) << v[i][cols - 1] << endl;
    }
    return os;
}

template <typename T>
vector<T> apply(double (*func)(double), vector<T> v)
{
    /* Applies the received function <func> to all elements in
     * vector <v>, and returns the resulting vector. */
    vector<T> result;
    for (auto &element : v)
        result.push_back(func(element));
    return result;
}

template <typename T, typename Op>
vector<T> vApply(vector<T> v1, vector<T> v2, Op f)
{
    /* Applies the operator <f>, element-wise, to all elements
     * in vectors <v1>, <v2>.
     * A new vector stores the results and is returned. */
    if (v1.size() != v2.size())
        throw invalid_argument("Ambos vectores deben ser del mismo tamaño");

    vector<T> result;
    for (int i = 0; i < v1.size(); i++)
        result.push_back(f(v1[i], v2[i]));
    return result;
}

template <typename T>
vector<vector<T>> operator*(vector<vector<T>> M, vector<vector<T>> N)
{
    /* Overloads multiplication operator to do a matrix multiplication using
     * matrices <M>, <N>. */
    if (M[0].size() != N.size())
        throw invalid_argument("La cantidad de columnas de M no coincide con la cantidad de filas de N");

    vector<vector<T>> result = {M.size(), vector<T>(N[0].size(), 0)};
    int counter = 0;
    for (int i = 0; i < M.size(); i++)
    {
        for (int j = 0; j < N[0].size(); j++)
        {
            vector<T> tmp;
            for (int k = 0; k < N.size(); k++)
            {
                tmp.push_back(M[i][k] * N[k][j]);
            }
            result[i][j] = reduce(tmp.begin(), tmp.end());
        }
    }
    return result;
}

double myPow(double base, double exp)
{
    /* Simulates the "power" operation.
     * This was needed because, for some reason, the built-in pow() function
     * couldn't be passed as an argument to vApply(). */
    return pow(base, exp);
}

template <typename T>
T func(T num)
{
    return 3 * pow(num, 2) + 2 * num - 6;
}

void e1()
{
    cout << "Hello, World!" << endl;
}

void e2()
{

    /* Crear las variables 𝑥 = 1 e 𝑦, y evaluar la siguiente función:
    𝑦 = 𝑓(𝑥) = 3𝑥^2 + 2𝑥 − 6.
    Despliegueen pantalla el resultado. */

    int x = 1;

    int y = func(x);
    std::cout << y << "\n";
}

void e3()
{
    /* Crear el vector 𝑥 = [−2, −1, 0, 1, 2] y evaluar la siguiente
    función: 𝑦 = 𝑓(𝑥) = 3𝑥2 + 2𝑥 − 6 .Despliegue en pantalla el
    resultado */

    vector<int> x = range(-2, 2, 1);

    std::cout << apply(func, x) << "\n";
}

void e4()
{
    /* Crear el vector 𝑥 = [−2, −1.9, −1.8, ... , 2], notar que
    el primer elemento del vector es -2 y el últimoes 2, con
    lo cual se deduce que hay una separación de 0.1 entre los
    elementos del vector. Evaluar la siguiente función:
    𝑦 = 𝑓(𝑥) = 3𝑥2 + 2𝑥 − 6 y desplegar los resultados.
    Despliegue en pantalla elresultado numérico. */

    vector<double> x = range(-2.0, 2.0, 0.1);

    std::cout << apply(func, x) << "\n";
}

void e5()
{
    /*
    ** Con x = [-2, -1.9, -1.8, ..., 2] evaluar las funciones:
    ** - y = sin x
    ** - y = log10(x)
    ** - y = e^x
    */
    vector<double> v = range(-2.0, 2.1, 0.1);

    int starting_idx = 0;
    for (; v[starting_idx] <= 0; starting_idx++)
        ;
    vector<double> vpos = vector(v.begin() + starting_idx, v.end());

    vector<double> vsin = apply(sin, v);

    // Logarithms can only accept positive numbers
    vector<double> vlog10 = apply(log10, vpos);
    for (int i = 0; i < vlog10.size() - v.size(); i++)
        vlog10.insert(vlog10.begin(), numeric_limits<double>::min());

    vector<double> vex = apply(exp, v);

    printf("|%9s%4s%9s%3s%11s%3s%8s%4s\n", "x", "|", "sin(x)", "|", "log10(x)", "|", "exp(x)", "|");
    for (int i = 0; i < v.size(); i++)
    {
        if (vlog10[i] == numeric_limits<double>::min())
            printf("|%9f%4s%9f%3s%11s%3s%8f%4s\n", v[i], "|", vsin[i], "|", "undef", "|", vex[i], "|");
        else
            printf("|%9f%4s%9f%3s%11f%3s%8f%4s\n", v[i], "|", vsin[i], "|", vlog10[i], "|", vex[i], "|");
    }
}

void e6()
{
    /* Crear los vectores
     * v1 = [1, 2, 3, 4]
     * v2 = [2, 4, 6, 8]
     * Realizar y mostrar el resultado de las siguientes operaciones:
     * v1*v2
     * v1/v2
     * v1^2
     * v1+v2
     * v1-v2
     * */

    vector<double> v1 = range(1.0, 4.0, 1.0);
    vector<double> v2 = range(2.0, 8.0, 2.0);
    vector<double> twos(4, 2);
    vector<double> vprod = vApply(v1, v2, multiplies());
    vector<double> vdiv = vApply(v1, v2, divides());
    vector<double> vpow = vApply(v1, twos, myPow);
    vector<double> vplus = vApply(v1, v2, plus());
    vector<double> vminus = vApply(v1, v2, minus());

    printf("|%4s%4s%4s%4s%4s%4s%6s%4s%10s%4s%4s%4s%4s%4s\n", "v1", "|", "v2", "|", "*", "|", "/", "|", "^", "|", "+", "|", "-", "|");
    for (int i = 0; i < v1.size(); i++)
        printf(
            "|%4.0f%4s%4.0f%4s%4.0f%4s%6.2f%4s%10.0f%4s%4.0f%4s%4.0f%4s\n",
            v1[i], "|",
            v2[i], "|",
            vprod[i], "|",
            vdiv[i], "|",
            vpow[i], "|",
            vplus[i], "|",
            vminus[i], "|");
}

void e7()
{
    /* Crear las matrices M, N. Realizar la multiplicación de matrices L = MN */
    vector<vector<int>> M = {
        {4, 2, 0, -1},
        {3, 1, -4, 0},
        {-1, 0, 3, 6}};

    vector<vector<int>> N = {
        {-1, 0},
        {2, 3},
        {-2, 1},
        {0, -1}};

    cout << M * N << endl;
}

void e8(){
    cout << "Hola Ivan" << endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
        throw invalid_argument("No se indicaron ejercicios a ejecutar.");

    for (int i = 1; i < argc; i++)
    {
        switch (stoi(argv[i]))
        {
        case 1:
            e1();
            break;
        case 2:
            e2();
            break;
        case 3:
            e3();
            break;
        case 4:
            e4();
            break;
        case 5:
            e5();
            break;
        case 6:
            e6();
            break;
        case 7:
            e7();
            break;
        case 8 :
            e8();
            break;
        // case 9 :
        //     e9();
        //     break;
        // case 10 :
        //     e10();
        //     break;
        default:
            cout << "Número de ejercicio inválido" << endl;
        }
    }

    return 0;
}
