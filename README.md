# bayespam

[![Build Status](https://travis-ci.com/zenoxygen/bayespam.svg?branch=master)](https://travis-ci.com/zenoxygen/bayespam)
[![Crates.io](https://img.shields.io/crates/v/bayespam.svg)](https://crates.io/crates/bayespam)
[![Docs](https://docs.rs/bayespam/badge.svg)](https://docs.rs/bayespam)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A simple bayesian spam classifier.

## About

Bayesam is inspired by [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering), a popular statistical technique of e-mail filtering.

Here, the message to be identified is cut into simple words, also called tokens.
That are compared to all the corpus of messages (spam or not), to determine the frequency of different tokens in both categories.

A probabilistic formula is used to calculate the probability that the message is spam or not.
When the probability is high enough, the bayesian system categorizes the message as spam.
Otherwise, he lets it pass. The probability threshold is fixed at 0.8 by default.

## Documentation

Learn more about Bayespam here: [https://docs.rs/bayespam](https://docs.rs/bayespam).

## Usage

Add to your `Cargo.toml`:

```ini
[dependencies]
bayespam = "0.3.0"
```

### Use the pre-trained model provided

```rust
extern crate bayespam;

use bayespam::classifier;

fn main() -> Result<(), std::io::Error> {
    // Identify a typical spam message
    let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
    let score = classifier::score(spam)?;
    let is_spam = classifier::identify(spam)?;
    println!("{:.4?}", score);
    println!("{:?}", is_spam);

    // Identify a typical ham message
    let ham = "Hi Bob, can you send me your machine learning homework?";
    let score = classifier::score(ham)?;
    let is_spam = classifier::identify(ham)?;
    println!("{:.4?}", score);
    println!("{:?}", is_spam);

    Ok(())
}
```

```bash
$> cargo run
0.9999
true
0.0604
false
```

### Train your own model and save it as JSON into a file

```rust
extern crate bayespam;

use bayespam::classifier::Classifier;
use std::fs::File;

fn main() -> Result<(), std::io::Error> {
    // Create a new classifier with an empty model
    let mut classifier = Classifier::new();

    // Train the classifier with a new spam example
    let spam = "Don't forget our special promotion: -30% on men shoes, only today!";
    classifier.train_spam(spam);

    // Train the classifier with a new ham example
    let ham = "Hi Bob, don't forget our meeting today at 4pm.";
    classifier.train_ham(ham);

    // Identify a typical spam message
    let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
    let score = classifier.score(spam);
    let is_spam = classifier.identify(spam);
    println!("{:.4}", score);
    println!("{}", is_spam);

    // Identify a typical ham message
    let ham = "Hi Bob, can you send me your machine learning homework?";
    let score = classifier.score(ham);
    let is_spam = classifier.identify(ham);
    println!("{:.4}", score);
    println!("{}", is_spam);

    // Serialize the model and save it as JSON into a file
    let mut file = File::create("my_super_model.json")?;
    classifier.save(&mut file, false)?;

    Ok(())
}
```

```bash
$> cargo run
0.9851
true
0.0100
false
```

```bash
$> cat my_super_model.json
{"token_table":{"forget":{"ham":1,"spam":1},"only":{"ham":0,"spam":1},"meeting":{"ham":1,"spam":0},"our":{"ham":1,"spam":1},"dont":{"ham":1,"spam":1},"bob":{"ham":1,"spam":0},"men":{"ham":0,"spam":1},"today":{"ham":1,"spam":1},"shoes":{"ham":0,"spam":1},"special":{"ham":0,"spam":1},"promotion:":{"ham":0,"spam":1}}}
```

## Contribution

Contributions via issues or pull requests are appreciated.

## License

Bayespam is distributed under the terms of the [MIT License](LICENSE).
