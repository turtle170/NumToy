#include "numtoy.hpp"
#include <iostream>
#include <cassert>

int main() {
    try {
        std::cout << "Initializing C++ NumToy Engine..." << std::endl;
        numtoy::Engine engine;

        std::cout << "Creating variables..." << std::endl;
        std::vector<double> x_vals = {1.0, 2.0, 3.0};
        std::vector<double> y_vals = {4.0, 5.0, 6.0};

        numtoy::Expr x = numtoy::Expr::var(engine, 1, "x", numtoy::DataType::Float, 32, x_vals);
        numtoy::Expr y = numtoy::Expr::var(engine, 2, "y", numtoy::DataType::Float, 32, y_vals);

        std::cout << "Building expression tree: z = (x + y) * 2.0" << std::endl;
        numtoy::Expr c = numtoy::Expr::constant(2.0);
        numtoy::Expr z = (x + y) * c;

        std::cout << "Compiling & Executing expression tree..." << std::endl;
        numtoy::Expr result = z.execute(engine);

        std::cout << "Unpacking results..." << std::endl;
        std::vector<double> results = result.unpack(engine, 3);

        std::cout << "Results: ";
        for (double v : results) {
            std::cout << v << " ";
        }
        std::cout << std::endl;

        assert(results.size() == 3);
        assert(results[0] == 10.0);
        assert(results[1] == 14.0);
        assert(results[2] == 18.0);

        std::cout << "ALL C++ WRAPPER TESTS PASSED SUCCESSFULLY!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
