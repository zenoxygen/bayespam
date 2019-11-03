//! # bayespam
//!
//! A simple bayesian spam classifier.
//!
//! ## About
//!
//! Bayespam is inspired by [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering), a popular statistical technique of e-mail filtering.
//!
//! Here, the message to be identified is cut into simple words, also called tokens.
//! That are compared to all the corpus of messages (spam or not), to determine the frequency of different tokens in both categories.
//!
//! A probabilistic formula is used to calculate the probability that the message is spam or not.
//! When the probability is high enough, the bayesian system categorizes the message as spam.
//! Otherwise, he lets it pass. The probability threshold is fixed at 0.8 by default.
//!
//! ## Usage
//!
//! Add to your `Cargo.toml`:
//!
//! ```ini
//! [dependencies]
//! bayespam = "0.2.0"
//! ```
//!
//! And to your crate root:
//!
//! ```
//! extern crate bayespam;
//!
//! use bayespam::classifier::Classifier;
//! ```
//!
//! ### Use the pre-trained model provided
//!
//! ```
//! extern crate bayespam;
//!
//! use bayespam::classifier;
//!
//! fn main() -> Result<(), std::io::Error> {
//!     // Identify a typical spam message
//!     let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
//!     let is_spam = classifier::identify(spam)?;
//!     assert!(is_spam);
//!
//!     // Identify a typical ham message
//!     let ham = "Hi Bob, can you send me your machine learning homework?";
//!     let is_spam = classifier::identify(ham)?;
//!     assert!(!is_spam);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Train your own model
//!
//! ```
//! extern crate bayespam;
//!
//! use bayespam::classifier::Classifier;
//!
//! fn main() {
//!     // Create a new classifier with an empty model
//!     let mut classifier = Classifier::new();
//!
//!     // Train the classifier with a new spam example
//!     let spam = "Don't forget our special promotion: -30% on men shoes, only today!";
//!     classifier.train_spam(spam);
//!
//!     // Train the classifier with a new ham example
//!     let ham = "Hi Bob, don't forget our meeting today at 4pm.";
//!     classifier.train_ham(ham);
//!
//!     // Identify a typical spam message
//!     let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
//!     let is_spam = classifier.identify(spam);
//!     assert!(is_spam);
//!
//!     // Identify a typical ham message
//!     let ham = "Hi Bob, can you send me your machine learning homework?";
//!     let is_spam = classifier.identify(ham);
//!     assert!(!is_spam);
//! }
//! ```

pub mod classifier;
