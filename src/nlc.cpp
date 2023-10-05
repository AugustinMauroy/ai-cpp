#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <string>
#include <sstream>

class NaturalLanguageClassifier {
public:
    struct Document {
        std::string document;
        std::string category;
    };

    NaturalLanguageClassifier() {
        // Initialize the classifier with empty data structures
    }

    void addDocument(const std::string& document, const std::string& category) {
        documents.push_back({document, category});
        categories.insert(category);

        std::istringstream iss(document);
        std::string word;
        while (iss >> word) {
            features.insert(word);
            const std::string featureCountKey = word + ":" + category;
            featureCount[featureCountKey]++;
        }
        categoryCount[category]++;
    }

    void train() {
        for (const std::string& category : categories) {
            const std::vector<Document>& documentsInCategory = getDocumentsInCategory(category);
            const std::vector<Document>& documentsNotInCategory = getDocumentsNotInCategory(category);
            const int featureCountInCategory = countFeaturesInCategory(category);
            const int featureCountNotInCategory = countFeaturesNotInCategory(category);
            const int totalDocumentsInCategory = documentsInCategory.size();
            const int totalDocumentsNotInCategory = documentsNotInCategory.size();

            for (const std::string& feature : features) {
                const std::string featureCountKey = feature + ":" + category;
                const int featureCountValue = featureCount[featureCountKey];
                const double featureProbInCategory = (static_cast<double>(featureCountValue) + 1) /
                                                     (featureCountInCategory + features.size());
                const double featureProbNotInCategory = (static_cast<double>(featureCountNotInCategory - featureCountValue) + 1) /
                                                         (totalDocumentsNotInCategory - featureCountInCategory + features.size());
                const double featureProb = featureProbInCategory / (featureProbInCategory + featureProbNotInCategory);
                const std::string featureProbKey = feature + ":" + category;
                featureProbabilities[featureProbKey] = featureProb;
            }
            const double categoryProb = static_cast<double>(totalDocumentsInCategory) / documents.size();
            categoryProbabilities[category] = categoryProb;
        }
    }

    std::vector<std::string> classify(const std::string& document, const std::string& context) {
        std::string processedInput = context + " " + document;
        std::transform(processedInput.begin(), processedInput.end(), processedInput.begin(), ::tolower);
        processedInput.erase(std::remove_if(processedInput.begin(), processedInput.end(), ::ispunct), processedInput.end());

        std::istringstream iss(processedInput);
        std::string word;
        std::vector<std::string> words;
        while (iss >> word) {
            words.push_back(word);
        }

        std::map<std::string, double> probabilities;
        for (const std::string& category : categories) {
            double prob = categoryProbabilities[category];
            for (const std::string& word : words) {
                const std::string featureProbKey = word + ":" + category;
                if (featureProbabilities.find(featureProbKey) != featureProbabilities.end()) {
                    prob *= featureProbabilities[featureProbKey];
                }
            }
            probabilities[category] = prob;
        }

        std::vector<std::pair<std::string, double>> sortedProbs(probabilities.begin(), probabilities.end());
        std::sort(sortedProbs.begin(), sortedProbs.end(),
                  [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                      return b.second < a.second;
                  });

        const double maxProb = sortedProbs[0].second;
        const double threshold = 0.1;
        std::vector<std::string> responseCategories;

        for (const auto& entry : sortedProbs) {
            if (entry.second >= maxProb * threshold) {
                responseCategories.push_back(entry.first);
            }
        }

        if (responseCategories.size() == 1 && maxProb < 0.5) {
            std::vector<std::string> ambiguousResponse;
            ambiguousResponse.push_back("I'm not sure. Can you please provide more context?");
            return ambiguousResponse;
        }

        return responseCategories;
    }

    int countFeaturesInCategory(const std::string& category) {
        int count = 0;
        for (const std::string& feature : features) {
            const std::string featureCountKey = feature + ":" + category;
            count += featureCount[featureCountKey];
        }
        return count;
    }

    int countFeaturesNotInCategory(const std::string& category) {
        int count = 0;
        for (const std::string& feature : features) {
            const std::string featureCountKey = feature + ":" + category;
            if (featureCount.find(featureCountKey) == featureCount.end()) {
                count += 1;
            }
        }
        return count;
    }

    void loadModelFromFile(const std::string& filePath) {
        std::ifstream file(filePath);
        if (file.is_open()) {
            // Load model data and update classifier parameters
            // ...
            file.close();
            train();  // Recalculate probabilities and other parameters as needed
        }
    }

    void saveModelToFile(const std::string& filePath) {
        // Save model data to a file
        // ...
    }

private:
    std::vector<Document> documents;
    std::set<std::string> features;
    std::set<std::string> categories;
    std::map<std::string, int> featureCount;
    std::map<std::string, double> featureProbabilities;
    std::map<std::string, int> categoryCount;
    std::map<std::string, double> categoryProbabilities;

    std::vector<Document> getDocumentsInCategory(const std::string& category) {
        std::vector<Document> result;
        for (const Document& doc : documents) {
            if (doc.category == category) {
                result.push_back(doc);
            }
        }
        return result;
    }

    std::vector<Document> getDocumentsNotInCategory(const std::string& category) {
        std::vector<Document> result;
        for (const Document& doc : documents) {
            if (doc.category != category) {
                result.push_back(doc);
            }
        }
        return result;
    }
};

