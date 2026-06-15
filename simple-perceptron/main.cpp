
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <random>
#include <iostream>

using namespace std;

struct Layer {
  vector<float> weights;
  float bias;
};

struct Sample {
  vector<float> features;
  int8_t variety; 

  Sample(vector<float> f, int8_t v) : features(f), variety(v) {}
};

vector<Sample> loadData(string dataset) {
  vector<Sample> data;
  map<string, int8_t> labelMap;
  int8_t varietyCounter = 0;

  ifstream file(dataset);

  if(!file.is_open()) {
    cerr << "Error opening file" << endl;
  }

  string header;
  getline(file, header);

  string line;
  while (getline(file,line)) {
    vector<float> features;
    string speciesName;
    int tokenIndex = 0;
    stringstream ss(line);
    string token;
    while (getline(ss, token, ',')) {
        if (tokenIndex < 4) {
          features.push_back(stof(token));
        } else {
          speciesName = token;
        }
        tokenIndex++;
    }

    if (!labelMap.count(speciesName)) {
      labelMap[speciesName] = varietyCounter++;
    }

    Sample sample(features, labelMap[speciesName]);
    data.push_back(sample);
  }


  return data;
}

float forward(vector<float> features, vector<Layer> layers) {
  vector<float> outputs;
  // calculate forward propagation (should we include activation function as part of the layer or as a common function here?)
  return 0.0f;
}

int main () {
  vector<Sample> data = loadData("iris.csv");
  mt19937 rng(time(0));
  uniform_real_distribution<float> dist(0.0f, 0.1f);
  
  Layer layer0;
  Layer layer1;
  Layer layer2;
  layer0.bias = 0.0f;
  layer0.weights.push_back(dist(rng));
  return 0;
}

// forward propagation

// loss calculation

// backpropagation

// update weights

// predict

// evaluation


