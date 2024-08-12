//  this header file contains functions to read CSV files into 2D arrays of proper shape.
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;


// ##################################
// Define the number of data points and features in the dataset 
int no_datapoints =  1960;
int no_features = 7;
string x_file = "x_mpg1960.csv";
string y_file = "y_mpg1960.csv";
// change the above values according to the dataset being used
// ##################################


vector<vector<double>> read_x(){
    int MAX_ROWS = no_datapoints;
    int MAX_COLS = no_features;

    // Open the CSV file
    ifstream file(x_file);
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return vector<vector<double>>();
    }

    // Define a 2D array to store the CSV data
    vector<vector<double>> data(MAX_ROWS, vector<double> (MAX_COLS));
    string line;
    int row = 0;
    // Store the CSV data from the CSV file to the 2D array
    while (getline(file, line) && row < MAX_ROWS) {
        stringstream ss(line);
        string cell;
        int col = 0;
        while (getline(ss, cell, ',') && col < MAX_COLS) {
            data[row][col] = stod(cell);
            col++;
        }
        row++;
    }
    // close the file after read opeartion is complete
    file.close();
    return data;
}

vector<vector<double>> read_y(){
    int MAX_ROWS = no_datapoints;
    int MAX_COLS = 1;
    // Open the CSV file
    ifstream file(y_file);
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return vector<vector<double>>();
    }

    // Define a 2D array to store the CSV data
    vector<vector<double>> data(MAX_ROWS, vector<double> (MAX_COLS));
    string line;
    int row = 0;
    // Store the CSV data from the CSV file to the 2D array
    while (getline(file, line) && row < MAX_ROWS) {
        stringstream ss(line);
        string cell;
        int col = 0;
        while (getline(ss, cell, ',') && col < MAX_COLS) {
            data[row][col] = stod(cell);
            col++;
        }
        row++;
    }
    // close the file after read opeartion is complete
    file.close();
    return data;
}