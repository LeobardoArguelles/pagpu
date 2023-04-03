// Program that checks if each file in "./test/9/puzzles/" has the same solution as the one in "./test/9/solutions/".
// The program will print the name of the file if the solution is different.
// The program will also print the name of the file if the solution is not found.
// If every file is correct, the program will print "All correct!".
// The program will also print the number of files that are correct and the number of files that are incorrect.
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

int main() {
    int correct = 0;
    int incorrect = 0;
    for (int i = 1; i <= 5000; i++) {
        string puzzle = "./test/9/my_solutions/solved_x9_" + to_string(i) + ".txt";
        string solution = "./test/9/solutions/9x9_" + to_string(i) + "_sol.txt";
        ifstream puzzleFile(puzzle);
        ifstream solutionFile(solution);
        vector<int> puzzleVector;
        vector<int> solutionVector;
        int number;

        if (!puzzleFile) {
            cout << "Puzzle file not found: " << i << endl;
            continue;
        }
        if (!solutionFile) {
            cout << "Solution file not found: " << i << endl;
            continue;
        }
        while (puzzleFile >> number) {
            puzzleVector.push_back(number);
        }
        while (solutionFile >> number) {
            solutionVector.push_back(number);
        }
        sort(puzzleVector.begin(), puzzleVector.end());
        sort(solutionVector.begin(), solutionVector.end());
        if (puzzleVector == solutionVector) {
            correct++;
        } else {
            incorrect++;
            cout << "Incorrect: " << i << endl;
        }
    }
    cout << "Correct: " << correct << endl;
    cout << "Incorrect: " << incorrect << endl;
    if (incorrect == 0) {
        cout << "All correct!" << endl;
    }
    return 0;
}
