
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <random>
#include <iostream>
#include <cmath>

using namespace std;

struct Layer {
  vector<vector<float>> weights;
  vector <float> biases;
  vector<float> z;
  vector<float> output;

  Layer(int numNeurons, int numInputs, mt19937& rng) {
    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    weights.resize(numNeurons, vector<float>(numInputs));
    biases.resize(numNeurons, 0.0f);
    z.resize(numNeurons, 0.0f);
    output.resize(numNeurons, 0.0f);
    for (auto& row: weights)
      for (auto& w : row)
        w = dist(rng);
  }
};

struct Sample {
  vector<float> features;
  int8_t variety; 

  Sample(vector<float> f, int8_t v) : features(f), variety(v) {}
};

struct Gradients {
  vector<vector<float>> dWeights;
  vector<float> dBiases;
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

vector<float> relu(vector<float> x) {
  for (auto& v : x)
    v = max(0.0f, v);
  return x;
}

float dRelu(float z) {
  if (z < 0)
    return 0;
  return 1;
}

vector<float> softmax(vector<float> x) {
  float sum = 0.0f;
  for (auto& v: x)
    sum+= exp(v);
  
  for (int i = 0; i < x.size(); i++) {
    x[i] = exp(x[i]) / sum;
  }
  return x;
}

vector<float> forward(vector<float> inputs, vector<Layer>& layers) {
  vector<float> current = inputs;
  for (int l = 0; l < layers.size() - 1; l++) {
    vector<float> next(layers[l].biases.size());
    for (int i = 0; i < layers[l].weights.size(); i++) {
      float sum = layers[l].biases[i];
      for (int j = 0; j < current.size(); j++)
        sum += layers[l].weights[i][j] * current[j];
      next[i] = sum;
      layers[l].z[i] = sum;
    }
    current = relu(next);
    layers[l].output = current;
  }

  vector<float> output(layers.back().biases.size());
  int l = layers.size() - 1;
  for (int i = 0; i < layers[l].weights.size(); i++) {
    float sum = layers[l].biases[i];
    for (int j = 0; j < current.size(); j++)
      sum += layers[l].weights[i][j] * current[j];
    output[i] = sum;
    layers[l].z[i] = sum;
  }
  vector<float> predictions = softmax(output);
  layers[l].output = predictions;
  return predictions;
}

float crossEntropy(vector<float> predictions, int correctClass) {
  return -log(predictions[correctClass]);
}




vector<Gradients> backward(vector<float> inputs, int correctClass, vector<Layer>& layers) {
  vector<float> truth(layers.back().output.size(), 0.0f);
  truth[correctClass] = 1.0f;
  vector<float> delta(layers.back().output.size());
  for (int i = 0; i < delta.size(); i++) {
    delta[i] = layers.back().output[i] - truth[i];
  }
  // add remaining gradients here
}



int main () {
  vector<Sample> data = loadData("iris.csv");
  mt19937 rng(time(0));
  Layer hidden(8, 4, rng);
  Layer output(3, 8, rng);
  vector<Layer> layers = {hidden, output};
  return 0;
}




// update weights

// predict

// evaluation


