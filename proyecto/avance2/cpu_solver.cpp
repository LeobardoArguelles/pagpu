#include <iostream>
#include <time.h>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace std;


#define N 9
#define UNASSIGNED 0
#define dir "./test/rated/puzzles/"
#define MAX_TEST 100

/* A utility function to print grid */
void print(int arr[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << arr[i][j] << " ";
        cout << endl;
    }
}


/* Returns a boolean which indicates whether
an assigned entry in the specified row matches
the given number. */
bool UsedInRow(int grid[N][N], int row, int num)
{
    for (int col = 0; col < N; col++)
        if (grid[row][col] == num)
            return true;
    return false;
}

/* Returns a boolean which indicates whether
an assigned entry in the specified column
matches the given number. */
bool UsedInCol(int grid[N][N], int col, int num)
{
    for (int row = 0; row < N; row++)
        if (grid[row][col] == num)
            return true;
    return false;
}

/* Returns a boolean which indicates whether
an assigned entry within the specified 3x3 box
matches the given number. */
bool UsedInBox(int grid[N][N], int boxStartRow,
               int boxStartCol, int num)
{
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            if (grid[row + boxStartRow]
                    [col + boxStartCol] ==
                                       num)
                return true;
    return false;
}

/* Returns a boolean which indicates whether
it will be legal to assign num to the given
row, col location. */
bool isSafe(int grid[N][N], int row,
            int col, int num)
{
    /* Check if 'num' is not already placed in
    current row, current column
    and current 3x3 box */
    return !UsedInRow(grid, row, num)
           && !UsedInCol(grid, col, num)
           && !UsedInBox(grid, row - row % 3,
                         col - col % 3, num)
           && grid[row][col] == UNASSIGNED;
}

/* Searches the grid to find an entry that is
still unassigned. If found, the reference
parameters row, col will be set the location
that is unassigned, and true is returned.
If no unassigned entries remain, false is returned. */
bool FindUnassignedLocation(int grid[N][N],
                            int& row, int& col)
{
    for (row = 0; row < N; row++)
        for (col = 0; col < N; col++)
            if (grid[row][col] == UNASSIGNED)
                return true;
    return false;
}

/* Takes a partially filled-in grid and attempts
to assign values to all unassigned locations in
such a way to meet the requirements for
Sudoku solution (non-duplication across rows,
columns, and boxes) */
bool SolveSudoku(int grid[N][N])
{
    int row, col;

    // If there is no unassigned location,
    // we are done
    if (!FindUnassignedLocation(grid, row, col))
        // success!
        return true;

    // Consider digits 1 to 9
    for (int num = 1; num <= 9; num++)
    {

        // Check if looks promising
        if (isSafe(grid, row, col, num))
        {

            // Make tentative assignment
            grid[row][col] = num;

            // Return, if success
            if (SolveSudoku(grid))
                return true;

            // Failure, unmake & try again
            grid[row][col] = UNASSIGNED;
        }
    }

    // This triggers backtracking
    return false;
}


// Function that iterates over the directory ./test_cases and runs the program on each file
// measuring the time it takes to solve the sudoku, storing the results in a file called
// results.txt.
// The function first reads how many numbers are in the first line of the file, and stores
// the number in the variable "N". Then it creates a 2D array of size N*N and stores the
// numbers in the file in the array. Finally, it calls the SolveSudoku function and
// measures the time it takes to solve the sudoku.
// It then appends the time it took to solve the sudoku to the file with the results.
// The function takes as input the number of tests to run and a file name to store the results.
// The function returns nothing.
void runTests(int tests, string file_name) {
    // Open the file with the results
    ofstream results;
    // Open a new filed called "results<N>x<N>.txt" where <N> is the size of the sudoku
    results.open(file_name);

    int count = 0;
    double total_time = 0;

    // Iterate over the files in the directory ./test_cases
    for (const auto & entry : fs::directory_iterator(dir)) {
        // Get the file name
        string file_name = entry.path().string();

        // Open the file
        ifstream file;
        file.open(file_name);

        string line;

        // Create a 2D array of size N*N to store the unsolved sudoku
        // and a 2D array of size N*N to store the solved sudoku
        int grid[N][N];
        int solved_grid[N][N];

        count++;
        if (count > tests)
            break;
        cout << count << endl;

        // Read the numbers in the file and store them in the array
        int row = 0;
        while (getline(file, line)) {
            stringstream ss(line);
            int col = 0;
            while (ss.good()) {
                string substr;
                getline(ss, substr, ' ');
                grid[row][col] = stoi(substr);
                col++;
            }
            row++;
        }

        // Close the file
        file.close();

        // Solve the sudoku 10 times, measuring the time it takes to solve the sudoku
        // each time and storing the average time in the variable "average_time"
        // then write the average time to the file with the results
        double average_time = 0;
        for (int i = 0; i < 10; i++) {
            // Copy He original sudoku to the solved sudoku
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    solved_grid[i][j] = grid[i][j];

            // Start the timer
            auto start = chrono::high_resolution_clock::now();

            // Solve the sudoku or print no solution
            if (!SolveSudoku(solved_grid))
                cout << "No solution exists";

            // Stop the timer
            auto stop = chrono::high_resolution_clock::now();

            // Calculate the time it took to solve the sudoku
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

            // Add the time it took to solve the sudoku to the average time
            average_time += duration.count();

        }

        // Calculate the average time
        average_time /= 10;

        // Write the average time to the file with the results
        total_time += average_time;

        // Create a file in "./test/9/my_solutions/" with the same name as a substring of the file name
        // starting from the 18th character (the name of the file without the path)
        // but with the prefix "solved_" and write the solved sudoku to the file
        // ofstream solved_file;
        // string solved_file_name = "./test/" + to_string(N) + "/my_solutions/solved_" + file_name.substr(18);
        // solved_file.open(solved_file_name);
        // if (!solved_file.is_open()) {
        //     cout << "Error opening file" << endl;
        //     return;
        // }
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < N; j++) {
        //         solved_file << solved_grid[i][j];
        //         // Add a space for every number except the last one
        //         if (j != N - 1)
        //             solved_file << " ";
        //     }
        //     solved_file << endl;
        // }
        // solved_file.close();

    }

    // Add the average time to the file with the results indicating the file number
    results << "Average time for " << tests << " tests: " << total_time / tests << endl;
}

// Function that runs a test on a single file, measuring how long it takes to solve the sudoku
// and printing the time it took to solve the sudoku.
// It also writes the time it took to solve the sudoku to the file with the results.
// The function takes as input the file name, the file number and a file pointer to the file where the results are stored.
// The function returns nothing.
void profile(string file_name, int file_number, ofstream *results) {
    // Open the file
    ifstream file;
    file.open(file_name);

    string line;

    // Create a 2D array of size N*N to store the unsolved sudoku
    // and a 2D array of size N*N to store the solved sudoku
    int grid[N][N];
    int solved_grid[N][N];

    // Read the numbers in the file and store them in the array
    int row = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        int col = 0;
        while (ss.good()) {
            string substr;
            getline(ss, substr, ' ');
            grid[row][col] = stoi(substr);
            col++;
        }
        row++;
    }

    // Close the file
    file.close();

    // Solve the sudoku 10 times, measuring the time it takes to solve the sudoku
    // each time and storing the average time in the variable "average_time"
    // then write the average time to the file with the results
    double average_time = 0;
    for (int i = 0; i < 10; i++) {
        // Copy He original sudoku to the solved sudoku
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                solved_grid[i][j] = grid[i][j];

        // Start the timer
        auto start = chrono::high_resolution_clock::now();

        // Solve the sudoku or print no solution
        if (!SolveSudoku(solved_grid))
            cout << "No solution exists";

        // Stop the timer
        auto stop = chrono::high_resolution_clock::now();

        // Calculate the time it took to solve the sudoku
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

        // Add the time it took to solve the sudoku to the average time
        average_time += duration.count();

    }

    // Calculate the average time
    average_time /= 10;

    // Write the average time to the file with the results
    cout << "Average time for file #" << file_number << ": " << average_time << endl;

    // Write the average time to the file with the results in a comma separated format
    // with the file number and the average time
    if (!results->is_open()) {
        cout << "Error opening file" << endl;
        return;
    }
    *results << file_number << "," << average_time << endl;
}

// Driver Code
int main()
{
    // Create a vector with the number of the puzzles to run the tests on
    // from 0 to 3000000
    vector<int> puzzles;
    for (int i = 3000000-1000; i < 3000000; i++)
        puzzles.push_back(i);

    // File name to store the results
    string result_file_name = "./results.txt";
    // Open the file
    ofstream result_file;
    result_file.open(result_file_name);

    // Run the tests
    for (int i = 0; i < puzzles.size(); i++) {
        // Generate filename appending the number of the puzzle to the prefix
        // ./test/rated/puzzles/9x9_ and the suffix .txt
        string file_name = "./test/rated/puzzles/9x9_" + to_string(puzzles[i]) + ".txt";
        profile(file_name, puzzles[i], &result_file);

        // If i is a multiple of 1000, print the number of puzzles solved
        if (i % 1000 == 0)
            cout << "Solved " << i << " puzzles" << endl;
    }

    return 0;
}
