#include <iostream>
#include <vector>
#include <cmath>

#define _USE_MATH_DEFINES


struct Value {
    int value;
    int index;

    Value(auto value, auto index) : value(value), index(index) {}
    Value(auto value) : value(value), index(-1) {}

    auto assign_class(auto id) 
    {
        index = id;
    }
};

struct Class 
{
    int class_id;
    double std;
    double mean;
    double prior;

    vector<Value> values;
    vector<double> likelihoods;

    Class(auto id) : class_id(id)
    {
        values = vector<Value>();
        likelihoods = vector<double>();
    }

    auto calc_all_values() 
    {
        calc_mean();
        calc_standard_deviation();
        calc_likelihoods();

    }

    auto insert(auto v) 
    {
        values.push_back(Value(v));
    }

    auto insert(auto v, auto id) 
    {
        values.push_back(Value(v, id));
    }

    auto calc_mean() 
    {
        mean = 0;
        for (auto v : values) 
        {
            mean += v.value;
        }
        mean /= values.size();
    }

    auto calc_standard_deviation() 
    {
        for (auto &v : values) 
        {
            std += pow(v.value - mean, 2);
        }
        std /= values.size();
        std = sqrt(std);
    }

    auto calc_likelihoods() 
    {
        for (auto v : values) 
        { 
            likelihoods.push_back(1 / (std * sqrt(2*M_PI)) * (pow(M_E, -pow(v.value - mean, 2) / (2 * pow(std, 2)))));
        }
    }
};


int main()
{

    return 0;
}