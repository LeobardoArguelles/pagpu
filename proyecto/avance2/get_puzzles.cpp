// Program tha reads the first 5000 lines of the file "sudoku.csv" and processes each line to convert it into a
// valid sudoku puzzle entry.
// Each line of "sudoku.csv" consists of two strings of 81 digits each, separated by a comma.
// The first string is the puzzle, the second string is the solution.
// For each string, the first 9 digits are the first row, the next 9 digits are the second row, and so on.
// The digits go from 0 to 9, where 0 means that the cell is empty.
// The program converts each string into a 9x9 array of integers, and then stores the array in two files:
// "./test/9/puzzels/9x9_<N>.txt" and "./test/9/solutions/9x9_<N>_sol.txt", where <N> is the line number of the puzzle in the file "sudoku.csv".
// The program also prints the number of puzzles that it has processed so far.
// The program stops when it has processed 500 puzzles.
// The program assumes that the file "sudoku.csv" is in the same directory as the program.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

int main() {
    FILE *fp, *pp; // file pointer
    char *line = NULL; // pointer to a string
    size_t len = 0; // length of the string
    ssize_t read; // number of characters read by getline()
    int i, j, k, l; // loop variables
    int puzzle[9][9]; // array to store the puzzle
    int solution[9][9]; // array to store the solution
    int count = 0; // number of puzzles processed so far
    char filename[100]; // string to store the filename

    fp = fopen("./test/rated/sudokus.csv", "r"); // open the file "sudoku.csv" for reading
    if (fp == NULL) { // if the file could not be opened
        printf("Could not open file sudoku.csv");
        exit(EXIT_FAILURE);
    }


    // Read file backwards
    while ((read = getline(&line, &len, fp)) != -1) { // while there are lines to read
        count++; // increment the number of puzzles processed so far
        // If count is a multiple of 100000, print the number of puzzles processed so far
        if (count % 100000 == 0) {
            printf("Processed %d puzzles so far\n", count);
        }
        for (i = 0; i < 9; i++) { // for each row
            for (j = 0; j < 9; j++) { // for each column
                puzzle[i][j] = line[i * 9 + j] - '0';
                solution[i][j] = line[i * 9 + j + 82] - '0';
            }
        }
        sprintf(filename, "./test/rated/puzzles/9x9_%d.txt", count); // create the filename for the puzzle
        pp = fopen(filename, "w"); // open the file for writing
        if (pp == NULL) { // if the file could not be opened
            printf("Could not open file %s", filename);
            exit(EXIT_FAILURE);
        }
        for (i = 0; i < 9; i++) { // for each row
            for (j = 0; j < 9; j++) { // for each column
                fprintf(pp, "%d", puzzle[i][j]); // write the value of the cell at position (i, j) to the file
                // Add a space for every cell except the last one
                if (j != 8) {
                    fprintf(pp, " ");
                }
            }
            fprintf(pp, "\n");
        }

        fclose(pp); // close the file

        sprintf(filename, "./test/rated/solutions/9x9_%d_sol.txt", count); // create the filename for the solution
        pp = fopen(filename, "w"); // open the file for writing
        if (pp == NULL) { // if the file could not be opened
            printf("Could not open file %s", filename);
            exit(EXIT_FAILURE);
        }

        for (i = 0; i < 9; i++) { // for each row
            for (j = 0; j < 9; j++) { // for each column
                fprintf(pp, "%d", solution[i][j]); // write the value of the cell at position (i, j) to the file
                if (j != 8) {
                    fprintf(pp, " ");
                }
            }
            fprintf(pp, "\n");
        }

        fclose(pp); // close the file
    }

    fclose(fp); // close the file
    if (line) { // if the string was allocated
        free(line); // free the memory
    }
    return 0;
}
