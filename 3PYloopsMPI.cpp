/******************************************************************************
 * Yloop MPI Simulator - Three Particle Quantum System
 *
 * Description: Parallel implementation of Yloop algorithm for Worldline
 *              Monte Carlo simulations using MPI for three-particle quantum
 *              systems with soft-Coulomb and an effective potential
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
#include <limits>
#include <gsl/gsl_multifit.h>  // Requires GSL library

using namespace std;

/*************************************************************************
Define the constants that we will use
**************************************************************************/

// Number of dimensions
const int DIMENSIONS = 1;

// Number of points per loop                                                   
const int POINTS_PER_LOOP = 5000;

// Number of loops                                                  
const int LOOPS = 5000;

// Number of repetitions (we keep this to gather statistics)
const int REPETITIONS = 10;

// Mass of particle 1
double m_1 = 1.0;

// Mass of particle 2
double m_2 = 1.0;

// Mass of particle 3
double m_3 = 1.0;

// Parameter for the distance between particles
//double d = 0.25;
double L = 2.1;     // For effective Trion model

// Constant for the Coulomb potential
double alpha = 1.0;                 

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

// Vector for the loops (particle 3)
vector<vector<double>> q_3(POINTS_PER_LOOP + 1, vector<double>(DIMENSIONS,0.0));

// Vector for the trajectory (particle 3)
vector<vector<double>> x_3(POINTS_PER_LOOP + 1, vector<double>(DIMENSIONS, 0.0));

// Dirichlet boundary conditions for the initial and final point (particle 1)
vector<double> x_1Initial = {0.0};
vector<double> x_1Final = {0.0};

// Dirichlet boundary conditions for the initial and final point (particle 2)
vector<double> x_2Initial = {0.0};
vector<double> x_2Final = {0.0};

// Dirichlet boundary conditions for the initial and final point (particle 3)
vector<double> x_3Initial = {0.0};
vector<double> x_3Final = {0.0};

// Vector for Gaussian random numbers 1
vector<double> ww_1(POINTS_PER_LOOP + 1, 0.0);                                

// Vector for Gaussian random numbers 2
vector<double> ww_2(POINTS_PER_LOOP + 1, 0.0);

// Vector for Gaussian random numbers 3
vector<double> ww_3(POINTS_PER_LOOP + 1, 0.0);

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
tuple<double,double,double> distanceBetweenInitialAndFinalPoints() {
    double xy_1 = 0.0;
    double xy_2 = 0.0;
    double xy_3 = 0.0;

    for (int i = 0; i < DIMENSIONS; i++) {
        xy_1 += (x_1Final[i] - x_1Initial[i]) * (x_1Final[i] - x_1Initial[i]);
        xy_2 += (x_2Final[i] - x_2Initial[i]) * (x_2Final[i] - x_2Initial[i]);
        xy_3 += (x_3Final[i] - x_3Initial[i]) * (x_3Final[i] - x_3Initial[i]);
    }

    return make_tuple(xy_1, xy_2, xy_3);

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
 * @function	safe_exp
 * @abstract	Compute exponential with overflow/underflow protection.
 * @param		x	The exponent value.
 * @result		The exponential result, bounded to avoid numerical issues.
 *
 */

inline double safe_exp(double x) {
    static const double MAX_EXP = log(std::numeric_limits<double>::max());
    static const double MIN_EXP = log(std::numeric_limits<double>::min());
    
    if (x > MAX_EXP) {
        return std::numeric_limits<double>::max();
    }
    else if (x < MIN_EXP) {
        return 0.0;
    }
    return exp(x);
}

/*!
 *
 * @function	safe_exp_erfc
 * @abstract	Compute the safe exponential times complementary error function.
 * @discussion	
 * @param		x   
 * @param		L   The length scale.
 * @result		
 *
 */

/*!
 *
 * @function	safe_exp_erfc
 * @abstract	Compute exponential times complementary error function safely.
 * @param		x	The argument value.
 * @param		L	The length scale parameter.
 * @result      The result of the safe exponential times complementary error function		
 *              (exp(x^2/2L^2) * erfc(x/(Lsqrt(2)))), handled numerically.
 *
 */

inline double safe_exp_erfc(double x, double L) {
    const double arg = x / (L * sqrt(2.0));
    const double x_sq = x * x;
    const double denom = 2.0 * L * L;
    
    // For large arguments, use asymptotic expansion to avoid inf*0
    if (arg > 26.0) {  // erfc(26) ≈ 1e-293 is near underflow
        const double prefactor = sqrt(M_PI/2.0) / L;
        const double exp_term = safe_exp(x_sq / denom);
        return prefactor * erfc(arg) * exp_term;
    }
    return sqrt(M_PI/2.0) * erfc(arg) * safe_exp(x_sq / denom) / L;
}

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
    double sqrt_T_m3 = sqrt(T / m_3);

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
            
            // Particle 3
            if (j > 0 && j < POINTS_PER_LOOP) {
                ww_3[j] = Marsaglia(0.0, 1.0/sqrt(2.0), gen);
            }

            // Periodic Boundary Conditions
            if (j == 0) {
                q_3[0][k] = 0.0;
            } else if (j == 1) {
                q_3[1][k] = sqrt(2.0 / POINTS_PER_LOOP) * 
                            sqrt(double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * ww_3[j];
            } else if (j == POINTS_PER_LOOP) {
                q_3[POINTS_PER_LOOP][k] = 0.0;
            } else {
                q_3[j][k] = sqrt(2.0 / POINTS_PER_LOOP) * 
                            sqrt(double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * ww_3[j] + 
                            (double(POINTS_PER_LOOP - j) / 
                            (POINTS_PER_LOOP + 1.0 - j)) * q_3[j-1][k];
            }
            
            // Full trajectory
            x_3[j][k] = x_3Initial[k] + 
                        (x_3Final[k] - x_3Initial[k]) *
                        (double(j) / double(POINTS_PER_LOOP)) +
                        sqrt_T_m3 * q_3[j][k];
        }
        
        // We calculate the value of the potential in the j point and sum over. We sum to
        // the value of the line integral over the trajectory. The following is the integral
        // of the potential. 

        //HERE WE DEFINE THE INTERACTIONS.

        // 1D Trion (soft-Coulomb)
        /*INTEGRAL += - (alpha / sqrt(pow(x_1[j][0] - x_2[j][0],2) + pow(d,2)))
                    - (alpha / sqrt(pow(x_1[j][0] - x_3[j][0],2) + pow(d,2)))
                    + (alpha / sqrt(pow(x_2[j][0] - x_3[j][0],2) + pow(d,2)));*/

        // 1D Effective Trion
        INTEGRAL += safe_exp_erfc(abs(x_1[j][0] - x_2[j][0]), L)
           - safe_exp_erfc(abs(x_1[j][0] - x_3[j][0]), L)
           - safe_exp_erfc(abs(x_2[j][0] - x_3[j][0]), L);
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
    oss << "3pMPIYL" << "Nl" << LOOPS << "Np" << POINTS_PER_LOOP << "Ti" 
        << minT << "Tf" << maxT << "Rep" << REPETITIONS << "L"<< L << "1DEffectiveTrion_GroundEnergy.txt";
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
    auto [xy_1, xy_2, xy_3] = distanceBetweenInitialAndFinalPoints();

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
                    for (int k = 0; k < LOOPS_PER_SLAVE; k++) {
                        // Variable to calculate the argument of the exponential (int dt V(x(t))
                        double INTEGRAL = 0.0;

                        // YLOOPS
                        INTEGRAL = yLoops(T, INTEGRAL, gen);

                        // We sum to the value of the line integral over the trajectory (physical background)
                        double WilsonLine = safe_exp(-(T / POINTS_PER_LOOP) * INTEGRAL);

                        // We gather the statistics
                        local_sum += WilsonLine;
                        local_sum_sq += WilsonLine * WilsonLine;
                        local_count++;
                    }                    
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
            
            // Prefactor K_{0,i}. Note that we are using M_PI (that is Pi)
            double K_01 = sqrt(pow((m_1) / (2.0 * M_PI * T), DIMENSIONS)) * exp(-m_1 * xy_1 / (2.0 * T)); 
            double K_02 = sqrt(pow((m_2) / (2.0 * M_PI * T), DIMENSIONS)) * exp(-m_2 * xy_2 / (2.0 * T));
            double K_03 = sqrt(pow((m_3) / (2.0 * M_PI * T), DIMENSIONS)) * exp(-m_3 * xy_3 / (2.0 * T));

            // Final propagator and SEM
            double K = K_01 * K_02 * K_03 * Exp_mean;
            double K_SEM = K_01 * K_02 * K_03 * SEM;
        
            // Print the results
            ofs << fixed << setprecision(12) << T << "\t" << K << "\t" << -log(K) 
                << "\t"<< K_SEM << "\t" << K_SEM / K << endl;
            ofs.flush();

            // To keep track of the progress
            cout << "T= " << T << " (" << (T - minT) / (maxT - minT) * 100 << "%)" << endl;
        }
    }
    
    // Non-linear fit: -log(K) = E0 * T - a + b * log(T) and cleaning
    if (RANK == 0) {

        // To perform the non-linear fits without using Mathematica (OPTIONAL)
        // If you want to use Mathematica, just comment this part until the end of the non-linear fit section.

        // Read data from file
        vector<double> T_vals, logK_vals, logK_errs;
        ifstream infile(outputFileName);
        double T, K, logK, K_SEM, rel_err;
        while (infile >> T >> K >> logK >> K_SEM >> rel_err) {
            T_vals.push_back(T);
            logK_vals.push_back(logK);  // File already contains -log(K)
            logK_errs.push_back(K_SEM / K);  // σ_{-log(K)} = σ_K / K
        }
        infile.close();

        // Prepare data for GSL
        const size_t n = T_vals.size();
        gsl_matrix *X = gsl_matrix_alloc(n, 3);
        gsl_vector *y = gsl_vector_alloc(n);
        gsl_vector *w = gsl_vector_alloc(n);

        for (size_t i = 0; i < n; ++i) {
            gsl_matrix_set(X, i, 0, T_vals[i]);      // Column 1: T
            gsl_matrix_set(X, i, 1, 1.0);            // Column 2: Constant (-a)
            gsl_matrix_set(X, i, 2, log(T_vals[i])); // Column 3: log(T)
            gsl_vector_set(y, i, logK_vals[i]);      // y = -log(K(T))
            gsl_vector_set(w, i, 1.0 / (logK_errs[i] * logK_errs[i]));  // Weights = 1/σ²
        }

        // Perform unweighted fit
        gsl_vector *c = gsl_vector_alloc(3);
        gsl_matrix *cov = gsl_matrix_alloc(3, 3);
        double chisq;
        gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(n, 3);
        gsl_multifit_linear(X, y, c, cov, &chisq, work);

        // Extract results
        double E0 = gsl_vector_get(c, 0);
        double a = -gsl_vector_get(c, 1);
        double b = gsl_vector_get(c, 2);
        double E0_err = sqrt(gsl_matrix_get(cov, 0, 0));
        double a_err = sqrt(gsl_matrix_get(cov, 1, 1));
        double b_err = sqrt(gsl_matrix_get(cov, 2, 2));

        // Perform weighted least-squares fit
        gsl_vector *c_w = gsl_vector_alloc(3);  // Coefficients: [E0, -a, b]
        gsl_matrix *cov_w = gsl_matrix_alloc(3, 3);
        double chisq_w;
        gsl_multifit_linear_workspace *work_w = gsl_multifit_linear_alloc(n, 3);
        gsl_multifit_wlinear(X, w, y, c_w, cov_w, &chisq_w, work_w);

        // Extract results
        double E0_w = gsl_vector_get(c_w, 0);       // Ground state energy
        double a_w = -gsl_vector_get(c_w, 1);       // Constant offset
        double b_w = gsl_vector_get(c_w, 2);        // log(T) coefficient
        double E0_err_w = sqrt(gsl_matrix_get(cov_w, 0, 0));  // Error in E0

        // Print results
        cout << "\nNon-linear fit results:\n";
        cout << fixed << setprecision(12);
        cout << "E0 = " << E0 << " ± " << E0_err << endl;
        cout << "a = " << a << " ± " << a_err << endl;
        cout << "b = " << b << " ± " << b_err << endl;
        cout << "\n Weighted Non-linear fit results:\n";
        cout << fixed << setprecision(12) << "E0 = " << E0_w << " ± " << E0_err_w << endl;
        cout << fixed << setprecision(12) << "a = " << a_w << endl;
        cout << fixed << setprecision(12) << "b = " << b_w << endl;

        // Cleanup
        gsl_multifit_linear_free(work);
        gsl_matrix_free(X);
        gsl_vector_free(y);
        gsl_vector_free(c);
        gsl_matrix_free(cov);
        
        gsl_multifit_linear_free(work_w);
        gsl_vector_free(w);
        gsl_vector_free(c_w);
        gsl_matrix_free(cov_w);

        // Write fitted energy to file
        ofstream efile("FittedEnergy_NonLinear.txt");
        efile << "Non-linear fit\n";
        efile << fixed << setprecision(12) << "E0 = " << E0 << " ± " << E0_err << endl;
        efile << fixed << setprecision(12) << "a = " << a << endl;
        efile << fixed << setprecision(12) << "b = " << b << endl;
        efile << "Weighted non-linear fit\n";
        efile << fixed << setprecision(12) << "E0 = " << E0_w << " ± " << E0_err_w << endl;
        efile << fixed << setprecision(12) << "a = " << a_w << endl;
        efile << fixed << setprecision(12) << "b = " << b_w << endl;
        efile.close();

        ofs.close();
        double end = MPI_Wtime();
        cout << oss.str() << endl;
        cout << "The process took " << (end - start) << " seconds." << endl;
        cout << "Process done" <<endl;
    }
    
    MPI_Finalize();
    return 0;
}