use std::collections::HashMap;
use std::fs::File;

use serde::{Deserialize, Serialize};
use serde_json::{from_reader, to_writer, to_writer_pretty};

const DEFAULT_FILE_PATH: &str = "model.json";
const INIT_RATING: f32 = 0.4;
const SPAM_PROB_THRESHOLD: f32 = 0.8;

#[derive(Debug, Default, Serialize, Deserialize)]
struct Counter {
    ham: u32,
    spam: u32,
}

/// A model.
#[derive(Default, Debug, Serialize, Deserialize)]
struct Model {
    token_table: HashMap<String, Counter>,
}

/// A bayesian spam classifier.
#[derive(Debug, Default)]
pub struct Classifier {
    model: Model,
}

impl Model {
    /// Load a model from the given file
    fn new_from_file(file: &mut File) -> Self {
        // TODO: propper error handling!
        from_reader(file).expect("reading from file failed")
    }

    /// Save the model into the file.
    fn save(&self, file: &mut File, pretty: bool) {
        // TODO: propper error handling!
        if pretty {
            to_writer_pretty(file, &self)
        } else {
            to_writer(file, &self)
        }
        .expect("writing to file failed")
    }
}

impl Classifier {
    /// Build a new `Classifier` with an empty model
    pub fn new() -> Self {
        Default::default()
    }

    /// Build a new `Classifier` trained from a the given `file`.
    pub fn new_from_file(file: &mut File) -> Self {
        Classifier {
            model: Model::new_from_file(file),
        }
    }

    /// Return a list of strings which contains only alphabetic letters,
    /// and keep only the words with a length greater than 2.
    ///
    /// * `msg` - String. Represents the message.
    fn load_word_list(msg: &str) -> Vec<String> {
        msg.replace(
            |c: char| !(c.is_lowercase() || c.is_uppercase() || c.is_whitespace() || c == ':'),
            "",
        )
        .trim()
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() > 2)
        .collect()
    }

    /// Save the model in a file.
    ///
    /// * `file` - File. The file to write to
    /// * `pretty` - bool. Pretty printing or not
    pub fn save(&self, file: &mut File, pretty: bool) {
        self.model.save(file, pretty)
    }

    /// Train the model of the classifier with a spam.
    ///
    /// * `msg` - String. Represents the spam message.
    pub fn train_spam(&mut self, msg: &str) {
        for word in Self::load_word_list(msg) {
            let counter = self.model.token_table.entry(word).or_default();
            counter.spam += 1;
        }
    }

    /// Train the model of the classifier with a ham.
    ///
    /// * `msg` - String. Represents the ham message.
    pub fn train_ham(&mut self, msg: &str) {
        for word in Self::load_word_list(msg) {
            let counter = self.model.token_table.entry(word).or_default();
            counter.ham += 1;
        }
    }

    fn spam_total_count(&self) -> u32 {
        self.model.token_table.values().map(|x| x.spam).sum()
    }

    fn ham_total_count(&self) -> u32 {
        self.model.token_table.values().map(|x| x.ham).sum()
    }

    /// Calculate and return the spam score of the message.
    /// The higher the score, the stronger the liklihood that the message is a spam is.
    ///
    /// * `msg` - String. Represents the message to score.
    pub fn score(&self, msg: &str) -> f32 {
        let ratings: Vec<_> = Self::load_word_list(msg)
            .into_iter()
            .map(|word| {
                if let Some(counter) = self.model.token_table.get(&word) {
                    if counter.spam > 0 && counter.ham == 0 {
                        return 0.99;
                    } else if counter.spam == 0 && counter.ham > 0 {
                        return 0.01;
                    } else if self.spam_total_count() > 0 && self.ham_total_count() > 0 {
                        let ham_prob = (counter.ham as f32) / (self.ham_total_count() as f32);
                        let spam_prob = (counter.spam as f32) / (self.spam_total_count() as f32);
                        return (spam_prob / (ham_prob + spam_prob)).max(0.01);
                    }
                }
                INIT_RATING
            })
            .collect();

        let ratings = match ratings.len() {
            0 => return 0.0,
            x if x > 20 => {
                let length = ratings.len();
                let mut ratings = ratings;
                ratings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); //this is actually okay, because we check if the numerator or denominiator is zero
                [&ratings[..10], &ratings[length - 10..]].concat()
            }
            _ => ratings,
        };

        let product: f32 = ratings.iter().product();
        let alt_product: f32 = ratings.iter().map(|x| 1.0 - x).product();

        product / (product + alt_product)
    }

    /// Decide whether the message is a spam or not.
    ///
    /// * `msg` - String. Represents the message to classify.
    pub fn is_spam(&self, msg: &str) -> bool {
        self.score(msg) > SPAM_PROB_THRESHOLD
    }
}

/// Calculate and return the spam score of the message,
/// based on the pre-trained model.
///
/// * `msg` - String. Represents the message to score.
pub fn score(msg: &str) -> f32 {
    // TODO: propper error handling!
    let mut file = File::open(DEFAULT_FILE_PATH).expect("Failed to open file");
    Classifier::new_from_file(&mut file).score(msg)
}

/// Decide whether the message is a spam or not,
/// based on the pre-trained model.
///
/// * `msg` - String. Represents the message to classify.
pub fn is_spam(msg: &str) -> bool {
    score(msg) > SPAM_PROB_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_spam() {
        // Create a new classifier with an empty model
        let mut classifier = Classifier::new();

        // Train the model with a new spam example
        let spam = "Don't forget our special promotion: -30% on men shoes, only today!";
        classifier.train_spam(spam);

        // Train the model with a new ham example
        let ham = "Hi Bob, don't forget our meeting today at 4pm.";
        classifier.train_ham(ham);

        // Classify a typical spam message
        let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
        let is_spam = classifier.is_spam(spam);
        assert!(is_spam);

        // Classifiy a typical ham message
        let ham = "Hi Bob, can you send me your machine learning homework?";
        let is_spam = classifier.is_spam(ham);
        assert!(!is_spam);
    }
}
