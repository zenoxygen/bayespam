use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

static DEFAULT_FILE_PATH: &str = "model.json";
const INIT_RATING: f32 = 0.4;
const SPAM_PROB_THRESHOLD: f32 = 0.8;

/// A model.
#[derive(Serialize, Deserialize)]
struct Model {
    spam_count_total: u32,
    ham_count_total: u32,
    token_table: HashMap<String, [u32; 2]>,
}

/// A bayesian spam classifier.
pub struct Classifier {
    model: Model,
}

impl Model {
    ///
    /// Build a new `Model`.
    ///
    /// * `file_path` - String. Path to the model file.
    /// * `create_new` - Boolean. If `true`, load a custom pre-trained model.
    /// If `false`, create a new empty model.
    fn new(file_path: &str, create_new: bool) -> Self {
        if create_new {
            Model {
                spam_count_total: 0,
                ham_count_total: 0,
                token_table: HashMap::new(),
            }
        } else {
            let json_file_path = Path::new(file_path);
            let json_file = File::open(&json_file_path).expect("Could not open file.");
            let deserialized: Model =
                serde_json::from_reader(json_file).expect("Could not read data.");

            Model {
                spam_count_total: deserialized.spam_count_total,
                ham_count_total: deserialized.ham_count_total,
                token_table: deserialized.token_table,
            }
        }
    }

    ///
    /// Save the model in a file.
    ///
    /// * `file_path` - String. The path where to save the model.
    fn save(&self, file_path: &str) {
        let json_file_path = Path::new(file_path);
        let mut file = File::create(&json_file_path).expect("Could not create file.");
        let serialized = serde_json::to_string(&self).unwrap();

        file.write_all(serialized.as_bytes())
            .expect("Could not write data.");
    }
}

impl Classifier {
    ///
    /// Build a new `Classifier`.
    ///
    /// * `file_path` - String. Path to the model file.
    /// * `create_new` - Boolean. If `true`, load a custom pre-trained model.
    /// If `false`, create a new empty model.
    pub fn new(file_path: &str, create_new: bool) -> Self {
        Classifier {
            model: Model::new(file_path, create_new),
        }
    }

    ///
    /// Return a list of strings which contains only alphabetic letters,
    /// and keep only the words with a length greater than 2.
    ///
    /// * `msg` - String. Represents the message.
    fn get_word_list(&self, msg: &str) -> Vec<String> {
        let re = Regex::new(r"[^a-zA-Z\s:]").unwrap();
        let clean = re.replace_all(msg, "");
        return clean
            .trim()
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| s.len() > 2)
            .collect();
    }

    ///
    /// Save the model in a file.
    ///
    /// * `file_path` - String. The path where to save the model.
    pub fn save(&self, file_path: &str) {
        self.model.save(file_path);
    }

    ///
    /// Train the model of the classifier.
    ///
    /// * `msg` - String. Represents the message.
    /// * `is_spam` - Boolean. If `true`, train the message as a spam.
    /// If `false`, train the message as ham.
    pub fn train(&mut self, msg: &str, is_spam: bool) {
        for word in self.get_word_list(msg) {
            if is_spam {
                self.model.spam_count_total += 1;
            } else {
                self.model.ham_count_total += 1;
            }
            if self.model.token_table.contains_key(&word) {
                let counter = self.model.token_table.get_mut(&word).unwrap();
                if is_spam {
                    counter[1] += 1;
                } else {
                    counter[0] += 1;
                }
            } else {
                if is_spam {
                    let counter: [u32; 2] = [0, 1];
                    self.model.token_table.insert(word.to_string(), counter);
                } else {
                    let counter: [u32; 2] = [1, 0];
                    self.model.token_table.insert(word.to_string(), counter);
                }
            }
        }
    }

    ///
    /// Calculate and return the spam score of the message.
    /// The higher the score, the stronger the liklihood that the message is a spam is.
    ///
    /// * `msg` - String. Represents the message to score.
    pub fn score(&mut self, msg: &str) -> f32 {
        let hashes: Vec<String> = self.get_word_list(msg);
        let mut ratings: Vec<f32> = Vec::new();
        for h in hashes {
            let mut rating: f32 = INIT_RATING;
            if self.model.token_table.contains_key(&h) {
                let counter = self.model.token_table.get_mut(&h).unwrap();
                let ham_count = counter[0];
                let spam_count = counter[1];
                if spam_count > 0 && ham_count == 0 {
                    rating = 0.99;
                } else if spam_count == 0 && ham_count > 0 {
                    rating = 0.01;
                } else if self.model.spam_count_total > 0 && self.model.ham_count_total > 0 {
                    let ham_prob: f32 = ham_count as f32 / self.model.ham_count_total as f32;
                    let spam_prob: f32 = spam_count as f32 / self.model.spam_count_total as f32;
                    rating = spam_prob / (ham_prob + spam_prob);
                    if rating < 0.01 {
                        rating = 0.01;
                    }
                }
            }
            ratings.push(rating);
        }
        if ratings.len() == 0 {
            return 0.0;
        }
        if ratings.len() > 20 {
            let length = ratings.len();
            ratings.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ratings = [&ratings[..10], &ratings[length - 10..]].concat();
        }

        let mut product: f32 = 1.0;
        for i in &ratings {
            product = product * i;
        }

        let inv: Vec<f32> = ratings.iter().map(|x| 1.0 - x).collect();

        let mut alt_product: f32 = 1.0;
        for i in &inv {
            alt_product = alt_product * i;
        }
        return product / (product + alt_product);
    }

    ///
    /// Decide whether the message is a spam or not.
    ///
    /// * `msg` - String. Represents the message to classify.
    pub fn is_spam(&mut self, msg: &str) -> bool {
        return self.score(msg) > SPAM_PROB_THRESHOLD;
    }
}

///
/// Calculate and return the spam score of the message,
/// based on the pre-trained model.
///
/// * `msg` - String. Represents the message to score.
pub fn score(msg: &str) -> f32 {
    let mut classifier = Classifier::new(DEFAULT_FILE_PATH, false);
    return classifier.score(msg);
}

///
/// Decide whether the message is a spam or not,
/// based on the pre-trained model.
///
/// * `msg` - String. Represents the message to classify.
pub fn is_spam(msg: &str) -> bool {
    return score(msg) > SPAM_PROB_THRESHOLD;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_spam() {
        // Create a new classifier with an empty model
        let mut classifier = Classifier::new("my_super_model.json", true);

        // Train the model with a new spam example
        let spam =
            String::from("Don't forget our special promotion: -30% on men shoes, only today!");
        classifier.train(&spam, true);

        // Train the model with a new ham example
        let ham = String::from("Hi Bob, don't forget our meeting today at 4pm.");
        classifier.train(&ham, false);

        // Classify a typical spam message
        let m1 = String::from("Lose up to 19% weight. Special promotion on our new weightloss.");
        let is_spam: bool = classifier.is_spam(&m1);
        assert_eq!(is_spam, true);

        // Classifiy a typical ham message
        let m2 = String::from("Hi Bob, can you send me your machine learning homework?");
        let is_spam: bool = classifier.is_spam(&m2);
        assert_eq!(is_spam, false);
    }
}
