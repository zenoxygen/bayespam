//! # bayespam
//!
//! A simple bayesian spam classifier.
//!
//! ## About
//!
//! Bayespam is inspired from [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering), a popular statistical technique of e-mail filtering.
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
//! bayespam = "0.1.0"
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
//! fn main() {
//!     // Classify a typical spam message
//!     let m1 = String::from("Lose up to 19% weight. Special promotion on our new weightloss.");
//!     let is_spam: bool = classifier::is_spam(&m1);
//!     assert_eq!(is_spam, true);
//!
//!     // Classifiy a typical ham message
//!     let m2 = String::from("Hi Bob, can you send me your machine learning homework?");
//!     let is_spam: bool = classifier::is_spam(&m2);
//!     assert_eq!(is_spam, false);
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
//!     let mut classifier = Classifier::new("my_super_model.json", true);
//!
//!     // Train the model with a new spam example
//!     let spam = String::from("Don't forget our special promotion: -30% on men shoes, only today!");
//!     classifier.train(&spam, true);
//!
//!     // Train the model with a new ham example
//!     let ham = String::from("Hi Bob, don't forget our meeting today at 4pm.");
//!     classifier.train(&ham, false);
//!
//!     // Classify a typical spam message
//!     let m1 = String::from("Lose up to 19% weight. Special promotion on our new weightloss.");
//!     let is_spam: bool = classifier.is_spam(&m1);
//!     assert_eq!(is_spam, true);
//!
//!     // Classifiy a typical ham message
//!     let m2 = String::from("Hi Bob, can you send me your machine learning homework?");
//!     let is_spam: bool = classifier.is_spam(&m2);
//!     assert_eq!(is_spam, false);
//! }
//! ```

pub mod classifier;
