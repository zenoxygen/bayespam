use std::collections::HashMap;
use std::fs::File;
use std::io;

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
    /// Load a pre-trained model from the given file.
    ///
    /// * `file` - File. The file to read the pre-trained model from.
    fn new_from_pre_trained(file: &mut File) -> Result<Self, io::Error> {
        // Deserialize an instance of type `Model` from the file
        let pre_trained_model = from_reader(file)?;

        // Return the `Model`
        Ok(pre_trained_model)
    }

    /// Save the model into the given file.
    ///
    /// * `file` - File. The file to write to.
    /// * `pretty` - Boolean. Pretty-printed JSON or not.
    fn save(&self, file: &mut File, pretty: bool) -> Result<(), io::Error> {
        if pretty {
            // Serialize the `Model` as JSON into the file
            to_writer_pretty(file, &self)?;
        } else {
            // Serialize the `Model` as as pretty-printed JSON into the file
            to_writer(file, &self)?;
        }

        Ok(())
    }
}

impl Classifier {
    /// Build a new `Classifier` with an empty model.
    pub fn new() -> Self {
        Default::default()
    }

    /// Build a new `Classifier` with a pre-trained model.
    ///
    /// * `file` - File. The file to read the pre-trained model from.
    pub fn new_from_pre_trained(file: &mut File) -> Result<Self, io::Error> {
        match Model::new_from_pre_trained(file) {
            Ok(pre_trained_model) => Ok(Classifier {
                model: pre_trained_model,
            }),
            Err(e) => Err(e),
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

    /// Save the model into the given file.
    ///
    /// * `file` - File. The file to write to.
    /// * `pretty` - Boolean. Pretty-printed JSON or not.
    pub fn save(&self, file: &mut File, pretty: bool) -> Result<(), io::Error> {
        self.model.save(file, pretty)?;

        Ok(())
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

    /// Return the total number of spam in token table.
    fn spam_total_count(&self) -> u32 {
        self.model.token_table.values().map(|x| x.spam).sum()
    }

    /// Return the total number of ham in token table.
    fn ham_total_count(&self) -> u32 {
        self.model.token_table.values().map(|x| x.ham).sum()
    }

    /// Calculate and return for each word the probability that it is part of a spam.
    ///
    /// * `msg` - String. Represents the message to score.
    fn rate_words(&self, msg: &str) -> Vec<f32> {
        Self::load_word_list(msg)
            .into_iter()
            .map(|word| {
                // If word was previously added in the model
                if let Some(counter) = self.model.token_table.get(&word) {
                    // If the word has only been part of spam messages,
                    // assign it a probability of 0.99 to be part of a spam
                    if counter.spam > 0 && counter.ham == 0 {
                        return 0.99;
                    // If the word has only been part of ham messages,
                    // assign it a probability of 0.01 to be part of a spam
                    } else if counter.spam == 0 && counter.ham > 0 {
                        return 0.01;
                    // If the word has been part of both spam and ham messages,
                    // calculate the probability to be part of a spam
                    } else if self.spam_total_count() > 0 && self.ham_total_count() > 0 {
                        let ham_prob = (counter.ham as f32) / (self.ham_total_count() as f32);
                        let spam_prob = (counter.spam as f32) / (self.spam_total_count() as f32);
                        return (spam_prob / (ham_prob + spam_prob)).max(0.01);
                    }
                }
                // If word was never added to the model,
                // assign it an initial probability to be part of a spam
                INIT_RATING
            })
            .collect()
    }

    /// Calculate and return the spam score of the message.
    /// The higher the score, the stronger the liklihood that the message is a spam is.
    ///
    /// * `msg` - String. Represents the message to score.
    pub fn score(&self, msg: &str) -> f32 {
        // Calculate for each word the probability that it is part of a spam
        let ratings = self.rate_words(msg);

        // If there are no ratings, return a score of 0
        // If there are more than 20 ratings, keep only the 10 first
        // and 10 last ratings to calculate a score
        // In all other cases, keep ratings to calculate a score
        let ratings = match ratings.len() {
            0 => return 0.0,
            x if x > 20 => {
                let length = ratings.len();
                let mut ratings = ratings;
                ratings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                [&ratings[..10], &ratings[length - 10..]].concat()
            }
            _ => ratings,
        };

        // Calculate the final score of the message to be a spam,
        // by multiplying all word ratings together
        let product: f32 = ratings.iter().product();
        let alt_product: f32 = ratings.iter().map(|x| 1.0 - x).product();
        product / (product + alt_product)
    }

    /// Identify whether the message is a spam or not.
    ///
    /// * `msg` - String. Represents the message to identify.
    pub fn identify(&self, msg: &str) -> bool {
        self.score(msg) > SPAM_PROB_THRESHOLD
    }
}

/// Calculate and return the spam score of the message, based on the pre-trained model.
/// The higher the score, the stronger the liklihood that the message is a spam is.
///
/// * `msg` - String. Represents the message to score.
pub fn score(msg: &str) -> Result<f32, io::Error> {
    let mut f = match File::open(DEFAULT_FILE_PATH) {
        Ok(file) => file,
        Err(e) => return Err(e),
    };

    match Classifier::new_from_pre_trained(&mut f) {
        Ok(classifier) => Ok(classifier.score(msg)),
        Err(e) => Err(e),
    }
}

/// Identify whether the message is a spam or not, based on the pre-trained model.
///
/// * `msg` - String. Represents the message to identify.
pub fn identify(msg: &str) -> Result<bool, io::Error> {
    let score = score(msg)?;
    let is_spam = score > SPAM_PROB_THRESHOLD;

    Ok(is_spam)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        // Create a new classifier with an empty model
        let mut classifier = Classifier::new();

        // Train the model with a new spam example
        let spam = "Don't forget our special promotion: -30% on men shoes, only today!";
        classifier.train_spam(spam);

        // Train the model with a new ham example
        let ham = "Hi Bob, don't forget our meeting today at 4pm.";
        classifier.train_ham(ham);

        // Identify a typical spam message
        let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
        let is_spam = classifier.identify(spam);
        assert!(is_spam);

        // Identify a typical ham message
        let ham = "Hi Bob, can you send me your machine learning homework?";
        let is_spam = classifier.identify(ham);
        assert!(!is_spam);
    }

    #[test]
    fn test_new_from_pre_trained() -> Result<(), io::Error> {
        // Identify a typical spam message
        let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
        let is_spam = identify(spam)?;
        assert!(is_spam);

        // Identify a typical ham message
        let ham = "Hi Bob, can you send me your machine learning homework?";
        let is_spam = identify(ham)?;
        assert!(!is_spam);

        Ok(())
    }
}
