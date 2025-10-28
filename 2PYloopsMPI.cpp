/******************************************************************************
 * Yloop MPI Simulator - Two Particle Quantum System
 * 
 * Description: Parallel implementation of Yloop algorithm for Worldline 
 *              Monte Carlo simulations using MPI for two-particle quantum
 *              systems with Coulomb and square well potentials.
 * 
 ******************************************************************************/

#define _USE_MATH_DEFINES  // Allows the use of mathematical constants,
                           // such as PI.

#include <cmath>    // For mathematical functions
#include <time.h>   // For time-related functions.
#include <chrono>   // For high-resolution clock and time point.

#include <random>   // For random number generation.
#include <vector>   // For dynamic arrays.
#include <fstream>  // For file I/O operations
#include <sstream>  // For string stream processing
#include <iomanip>  // For input/output manipulators 
#include <iostream> // For standard I/O operations
#include <tuple>    // For tuple
#include <mpi.h>    // For the MPI library

using namespace std;

/*************************************************************************
Define the constants that we will use
**************************************************************************/

// Number of dimensions
const int DIMENSIONS = 3;

// Number of points per loop                                                   
const int POINTS_PER_LOOP = 5000;

// Number of loops                                                  
const int LOOPS = 1000;

// Number of repetitions (we keep this to improve the statistics)
const int REPETITIONS = 100;

// Mass of particle 1
double m_1 = 1.0;

// Mass of particle 2
double m_2 = 1.0;

// Reduced mass of the system
double mu = (m_1 * m_2) / (m_1 + m_2);

// Parameter for the distance between particles
double d = 2.0;

// Constant for the Coulomb potential
double alpha = 1.0;                                      

// Potential depth for the square Well
double V_e = -0.25;

// Potential barrier for the square Well
double V_h = -0.25;                   

// Rectangular dimensions for the square Well (3D)
//double L_x = 1.0;
//double L_y = 1.0;
double L_z = 1.0;

// Definitions for the Time
double minT = 1.0;
double maxT = 30.0;
double deltaT = 1.0;

/*************************************************************************
Define the vectors that we will use
**************************************************************************/

// Vector for the loops (particle 1)
vector<vector<double>> q_1(POINTS_PER_LOOP + 1, vector<double>(DIMENSIONS,0.0));

// Vector for the trajectory (particle 1)
vector<vector<double>> x_1(POINTS_PER_LOOP + 1, vector<double>(DIMENSIONS, 0.0));

// Vector for the loops (particle 2)
vector<vector<double>> q_2(POINTS_PER_LOOP + 1, vector<double>(DIMENSIONS,0.0));

// Vector for the trajectory (particle 2)
vector<vector<double>> x_2(POINTS_PER_LOOP + 1, vector<double>(DIMENSIONS, 0.0));

// Dirichlet boundary conditions for the initial and final point (particle 1)
vector<double> x_1Initial = {0.01, 0.0, d/2};
vector<double> x_1Final = {0.01, 0.0, d/2};

// Dirichlet boundary conditions for the initial and final point (particle 2)
vector<double> x_2Initial = {0.0, 0.01, -d/2};
vector<double> x_2Final = {0.0, 0.01, -d/2};

// Vector for Gaussian random numbers 1
vector<double> ww_1(POINTS_PER_LOOP + 1, 0.0);                                

// Vector for Gaussian random numbers 2
vector<double> ww_2(POINTS_PER_LOOP + 1, 0.0);

/*************************************************************************
Define the functions that we will use
**************************************************************************/

/*!
 *
 * @function	fileExists
 * @abstract	Verifies if the file exists.
 * @param		filename	The name of the file.
 * @result		Whether the file exists or not.
 *
 */
bool fileExists(const std::string& filename) {
    std::ifstream file(filename.c_str());

    return file.good();
} // End of fileExists(filename)

/*!
 *
 * @function	distanceBetweenInitialAndFinalPoints
 * @abstract	Calculate the distance between initial and final points
 *              in the argument of the exponential function.
 * @result		The distance between initial and final points.
 *
 */
tuple<double,double> distanceBetweenInitialAndFinalPoints(void) {
    double xy_1 = 0.0;
    double xy_2 = 0.0;

    for (int i = 0; i < DIMENSIONS; i++) {
        xy_1 += (x_1Final[i] - x_1Initial[i]) * (x_1Final[i] - x_1Initial[i]);
        xy_2 += (x_2Final[i] - x_2Initial[i]) * (x_2Final[i] - x_2Initial[i]);
    }

    return make_tuple(xy_1, xy_2);

} // End of distanceBetweenInitialAndFinalPoints()

/*!
 * @function    Marsaglia
 * @abstract    Marsaglia polar method
 * @discussion  The Marsaglia polar method is a pseudo-random number sampling method for
 *              generating a pair of independent standard normal random variables. This
 *              type of variables is frequently used in computer science, and in
 *              particular, in applications of the Monte Carlo method.
 * @result      Return a double value using the mean and standard deviation.
 * 
 */
double Marsaglia(double mean, double stdDev, mt19937_64& gen) {

    /* Initialise the Mersenne Twister random number generator with the seed
    * defined.
    *
    * We are using mt19937_64, a Mersenne Twister, general-purpose,
    * pseudorandom number generator (PRNG) of 64-bit numbers with a state
    * size of 19937 bits. It is based on the Mersenne prime 2^19937 – 1. It
    * produces high quality, but not cryptographically secure, unsigned integer
    * random numbers and was designed specifically to rectify most of the flaws
    * found in older PRNGs.
    */
    
    /* Define a uniform distribution for random numbers.
    *
    * Get a random number in the range [-1.0,1.0) where all intervals of the
    * same length within it are equally probable. Use dis to transform the
    * random unsigned integer generated by gen into a double in [-1.0,1.0).
    * Each call to dis(gen) generates a new random double.
    */
    
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    double x, y, z;

    do {
        // Every call to dis(gen) generates a new random double.
        x = dis(gen);
        y = dis(gen);
        z = pow(x, 2) + pow(y, 2);
    } while (z >= 1.0 || z == 0.0);
    z = sqrt((-2.0 * log(z)) / z);

    return mean + stdDev * x * z;

} // End of Marsaglia(mean, stdDev)

/*!
 *
 * @function	YLOOPS
 * @abstract	Compute the YLOOPS.
 * @discussion	Implements the Y-loop algorithm to construct periodic Brownian quantum
 *		        fluctuations q for both particles and assemble the full trajectories x_1 and x_2.
 * @param		T	Time (the number of the current
 *				loop).
 * @param		INTEGRAL    The integral of the Wilson line (potential).
 * @param		gen    The UNIQUE seed per rank.
 * @result		The updated value of the integral
 *              of the potential.
 *
 */

double yLoops(double T, double INTEGRAL, mt19937_64& gen) {
    
    // Root factor for the q's
    double sqrt_T_m1 = sqrt(T / m_1);
    double sqrt_T_m2 = sqrt(T / m_2);

    // YLOOPS
    for (int j = 0; j <= POINTS_PER_LOOP; j++) {
        for (int k = 0; k < DIMENSIONS; k++) {
            // Particle 1
            if (j > 0 && j < POINTS_PER_LOOP) {
                ww_1[j] = Marsaglia(0.0, 1.0/sqrt(2.0), gen);
            }

            // Periodic Boundary Conditions
            if (j == 0) {
                q_1[0][k] = 0.0;
            } else if (j == 1) {
                q_1[1][k] = sqrt(2.0 / POINTS_PER_LOOP) * 
                            sqrt(double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * ww_1[j];
            } else if (j == POINTS_PER_LOOP) {
                q_1[POINTS_PER_LOOP][k] = 0.0;
            } else {
                q_1[j][k] = sqrt(2.0 / POINTS_PER_LOOP) * 
                            sqrt(double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * ww_1[j] + 
                            (double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * q_1[j-1][k];
            }
            
            // Full trajectory
            x_1[j][k] = x_1Initial[k] + 
                        (x_1Final[k] - x_1Initial[k]) *
                        (double(j) / double(POINTS_PER_LOOP)) +
                        sqrt_T_m1 * q_1[j][k];
            
            // Particle 2
            if (j > 0 && j < POINTS_PER_LOOP) {
                ww_2[j] = Marsaglia(0.0, 1.0/sqrt(2.0), gen);
            }

            // Periodic Boundary Conditions
            if (j == 0) {
                q_2[0][k] = 0.0;
            } else if (j == 1) {
                q_2[1][k] = sqrt(2.0 / POINTS_PER_LOOP) * 
                            sqrt(double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * ww_2[j];
            } else if (j == POINTS_PER_LOOP) {
                q_2[POINTS_PER_LOOP][k] = 0.0;
            } else {
                q_2[j][k] = sqrt(2.0 / POINTS_PER_LOOP) * 
                            sqrt(double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * ww_2[j] + 
                            (double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * q_2[j-1][k];
            }
            
            // Full trajectory
            x_2[j][k] = x_2Initial[k] + 
                        (x_2Final[k] - x_2Initial[k]) *
                        (double(j) / double(POINTS_PER_LOOP)) +
                        sqrt_T_m2 * q_2[j][k];
        }

        // We calculate the value of the potential in the j point and sum over. We sum to
        // the value of the line integral over the trajectory. The following is the integral
        // of the potential. 

        //HERE WE DEFINE THE INTERACTIONS.

        // 3D soft-Coulomb potential + 1D Square Well potential and barrier in Z direction (L width, V_e depth, V_h barrier altitute)
	    if (abs(x_1[j][2]) < (L_z / 2.0) && abs(x_2[j][2]) < (L_z / 2.0))
        {
            INTEGRAL += -(alpha / sqrt(pow(x_1[j][0] - x_2[j][0],2) + 
                      pow(x_1[j][1] - x_2[j][1],2) + pow(x_1[j][2] - x_2[j][2],2))) + V_e + V_h;
        } else if (abs(x_1[j][2]) < (L_z / 2.0))
        {
            INTEGRAL += -(alpha / sqrt(pow(x_1[j][0] - x_2[j][0],2) + 
                      pow(x_1[j][1] - x_2[j][1],2) + pow(x_1[j][2] - x_2[j][2],2))) + V_e;
        } else if (abs(x_2[j][2]) < (L_z / 2.0))
        {
            INTEGRAL += -(alpha / sqrt(pow(x_1[j][0] - x_2[j][0],2) + 
                      pow(x_1[j][1] - x_2[j][1],2) + pow(x_1[j][2] - x_2[j][2],2))) + V_h;
        } else
        {
            INTEGRAL += -(alpha / sqrt(pow(x_1[j][0] - x_2[j][0],2) + 
                      pow(x_1[j][1] - x_2[j][1],2) + pow(x_1[j][2] - x_2[j][2],2)));
        }
    } 

    return INTEGRAL;
} // End of yLoops(T, INTEGRAL, gen)

/**
  Main program
 **/

int main(int argc, char** argv) {


  
    MPI_Init(&argc, &argv);

    int RANK, SIZE;
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

    // Record start time
    double start = MPI_Wtime();

    // This is the name of the file where we’ll save the output.
    // If no name is provided by the user, we’ll use the
    // following default value.
    std::ostringstream oss;
    oss << "2pMPIYL" << "Nl" << LOOPS << "Np" << POINTS_PER_LOOP << "Ti" 
        << minT << "Tf" << maxT << "Rep" << REPETITIONS << "d"<< d << "V_hV_e" << abs(V_h) <<"3DsoftCoulombSqrWell_GroundEnergy.txt";
    std::string outputFileName = oss.str();

    // File handling (only Master)
    if (RANK == 0)
    {
        for (int i = 1; i < argc; i++) {
            // Retrieve the arguments from the command line!
            outputFileName.assign(argv[i]);

            // Search for the last occurrence of the output flag!
            if (outputFileName.rfind("--output=", 0) == 0) {

                // Determine the name for the file given by the user!
                outputFileName.assign(outputFileName.substr(
                    std::string("--output=").length()));

                if (outputFileName.empty()) {
                    std::cout << "Missing argument: " 
                                << "Name of the output file!" 
                                << std::endl;
                    return 0;
                } // End of if (outputFileName.empty())

                cout << outputFileName << std::endl;

                // Check if the file exists, in which we must notify the
                // user.
                if (fileExists(outputFileName)) {

                    std::string response;

                    std::cout << "File \"" << outputFileName 
                                << "\" already exists. Overwrite? (y/n): ";

                    std::getline(std::cin, response);

                    if (response != "y" && response != "Y") {
                        std::cout << "Aborting! File was not overwritten.\n";
                        return 0;
                    } // End of if (response != "y" && response != "Y")
                }
            } // End of if (outputFileName.rfind("--output="...
        } // End of for (int i = 1; i < argc; i++)
    }
    

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\\\//\\//\\\\//
    //
    // If we’ve reached this point, it means we’ve got an
    // acceptable name for the file where we’ll save the output.
    //
    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\\\//\\//\\\\//

    // Create an output stream to print out strings. Characters
    // can be inserted into the stream with any operation allowed
    // on output streams.
    //
    // Using ios::trunc guarantess that if the file is opened for
    // output operations and it already existed, its previous
    // content is deleted and replaced by the new one.

    std::ofstream ofs(outputFileName,
                            std::ios::trunc | std::ios::out);
    
    // Divide the total loops among the slaves
    int LOOPS_PER_SLAVE = LOOPS / SIZE;
    int REMAINDER = LOOPS % SIZE;
    
    if (RANK == 0) {
        cout << "Running on " << SIZE << " processes." << endl;
        cout << "Distributing " << LOOPS << " loops among " << SIZE << " processes." << endl;
    } 
    // Distribute the remainder among the first few processes
    if (RANK < REMAINDER) {
        LOOPS_PER_SLAVE++;
    }

    /* Generate a UNIQUE seed per rank based on the current time.
    *
    * Seeding a random number generator in C++ is the process of initialising
    * the generator with a starting value, called a seed. This seed value is
    * used to start the algorithm that generates random numbers. It ensures
    * that the sequence of random numbers can be reproduced if the same seed
    * is used again. Without seeding, the random number generator would
    * produce the same sequence every time the program runs, which would not
    * be truly random.
    */

    unsigned SEED = chrono::system_clock::now().time_since_epoch().count() + RANK;
    mt19937_64 gen(SEED);
	
    // Here we start the loop over the values of T. 
    // For each value of T, we generate an independent
    // set of unitary loops.

    // Calculate the distance between initial and final points.
    auto [xy_1, xy_2] = distanceBetweenInitialAndFinalPoints();

    // Build the loop cloud
    for(double T = minT; T <= maxT; T+= deltaT)
    {   
        // Define variables for the computation of K
        // Local variables for the statistics
        double local_sum = 0.0;
        double local_sum_sq = 0.0;
        long local_count = 0;

        // Here we initiate the for statement over the number of repetitions
        for (int R = 0; R < REPETITIONS; R++) {   
            // This is where we will generate the number of loops
            // (it will be different for each value of T).
            for(int i = 0; i < LOOPS_PER_SLAVE; i++) {
                for (int j = 0; j < LOOPS_PER_SLAVE; j++) {
                    // Variable to calculate the argument of the exponential (int dt V(x(t))
                    double INTEGRAL = 0.0;

                    // YLOOPS
                    INTEGRAL = yLoops(T, INTEGRAL, gen);

                    // We sum to the value of the line integral over the trajectory (physical background)
                    double WilsonLine = exp(-(T / POINTS_PER_LOOP) * INTEGRAL);

                    // We gather the statistics
                    local_sum += WilsonLine;
                    local_sum_sq += WilsonLine * WilsonLine;
                    local_count++;
                }
            } 	
        }// All loops generated for this value of T
        
        // Reduce results to Master
        double global_sum, global_sum_sq;
        long global_count;
        
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        // Master computes final results
        if (RANK == 0) {
            // Compute the global mean
            double Exp_mean = global_sum / global_count;

            // Compute the SEM and the variance
            double variance = (global_sum_sq - (global_sum * global_sum / global_count)) / (global_count - 1);
            double SEM = sqrt(variance / global_count);
            
            // Prefactor K_0. Note that we are using M_PI (that is Pi)
            double K_01 = sqrt(pow((m_1) / (2.0 * M_PI * T), DIMENSIONS)) * exp(-m_1 * xy_1 / (2.0 * T)); 
            double K_02 = sqrt(pow((m_2) / (2.0 * M_PI * T), DIMENSIONS)) * exp(-m_2 * xy_2 / (2.0 * T));

            // Final propagator and SEM
            double K = K_01 * K_02 * Exp_mean;
            double K_SEM = K_01 * K_02 * SEM;
        
            // Print the results
            ofs << fixed << setprecision(12) << T << "\t" << K << "\t" << -log(K) 
                << "\t"<< K_SEM << "\t" << K_SEM / K << endl;
            ofs.flush();

            // To keep track of the progress
            cout << "T= " << T << " (" << (T - minT) / (maxT - minT) * 100 << "%)" << endl;
        }
    }
    
    if (RANK == 0) {
        
	// To perform the linear fits without using Mathematica (OPTIONAL)
    // If you want to use Mathematica, just comment this part until the end of the linear fit section.
	
    // Read the output file (or use stored data)
        vector<double> T_vals, logK_vals, logK_errs;
        ifstream infile(outputFileName);
        double T, K, logK, K_SEM, rel_err;
        while (infile >> T >> K >> logK >> K_SEM >> rel_err) {
            T_vals.push_back(T);
            logK_vals.push_back(logK);  // Store -log(K)
            logK_errs.push_back(K_SEM / K);  // Error in -log(K) = rel. error in K
        }
        infile.close();

        // Simple linear regression (unweighted)
        const size_t n = T_vals.size();
	    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        
        for (size_t i = 0; i < T_vals.size(); ++i) {
            sum_x += T_vals[i];
            sum_y += logK_vals[i];
            sum_xy += T_vals[i] * logK_vals[i];
            sum_xx += T_vals[i] * T_vals[i];
        }

        double delta = n * sum_xx - sum_x * sum_x;
        double E0 = (n * sum_xy - sum_x * sum_y) / delta;
        double C = (sum_xx * sum_y - sum_x * sum_xy) / delta;

        // Error estimation
        double sse = 0;
        for (size_t i = 0; i < T_vals.size(); ++i) {
            double residual = logK_vals[i] - (E0 * T_vals[i] + C);
            sse += residual * residual;
        }
        double stdev = sqrt(sse / (n - 2));
        double E0_err = stdev * sqrt(n / delta);
        double C_err = stdev * sqrt(sum_xx / delta);

        // Perform a weighted linear fit: -log(K) = E0 * T + C
        double sum_w = 0.0, sum_wx = 0.0, sum_wy = 0.0, sum_wxy = 0.0, sum_wxx = 0.0;
        for (size_t i = 0; i < T_vals.size(); ++i) {
            double w = 1.0 / (logK_errs[i] * logK_errs[i]);  // Weight = 1/σ²
            sum_w += w;
            sum_wx += w * T_vals[i];
            sum_wy += w * logK_vals[i];
            sum_wxy += w * T_vals[i] * logK_vals[i];
            sum_wxx += w * T_vals[i] * T_vals[i];
        }

        // Solve for E0 (slope) and C (intercept)
        double delta_w = sum_w * sum_wxx - sum_wx * sum_wx;
        double E0_w = (sum_w * sum_wxy - sum_wx * sum_wy) / delta_w;
        double C_w = (sum_wxx * sum_wy - sum_wx * sum_wxy) / delta_w;

        // Estimate error in E0
        double E0_err_w = sqrt(sum_w / delta_w);

        // Print the fitted energy
        cout << fixed << setprecision(12) << "\nFitted Ground State Energy: E0 = " << E0 << " ± " << E0_err << endl;
        cout << fixed << setprecision(12) << "c = " << C << endl;
        cout << fixed << setprecision(12) << "\nWeighted Fitted Ground State Energy: E0 = " << E0_w << " ± " << E0_err_w << endl;
        cout << fixed << setprecision(12) << "c = " << C_w << endl;

        // Optional: Write fitted energy to a file
        ofstream efile("FittedEnergy.txt");
        efile << "Linear Fit (standard least squares)" << endl;
        efile << fixed << setprecision(12) << "E0 = " << E0 << " ± " << E0_err << std::endl;
        efile << fixed << setprecision(12) << "C = " << C << endl;
        efile << "Weighted Linear Fit (weighted least squares)" << endl;
        efile << fixed << setprecision(12) << "E0 = " << E0_w << " ± " << E0_err_w << std::endl;
        efile << fixed << setprecision(12) << "C = " << C_w << endl;
        efile.close();

        // Record end time and cleanup
        ofs.close();
        cout << outputFileName << std::endl;
        double end = MPI_Wtime();
        cout << "The process took " << (end - start) << " seconds." << endl;
    }
    
    MPI_Finalize();
    return 0;

}
