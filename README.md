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

## Usage

Add to your `Cargo.toml`:

```ini
[dependencies]
bayespam = "0.1.4"
```

### Use the pre-trained model provided

```rust
extern crate bayespam;

use bayespam::classifier;

fn main() {
    // Classify a typical spam message
    let m1 = String::from("Lose up to 19% weight. Special promotion on our new weightloss.");
    let score: f32 = classifier::score(&m1);
    let is_spam: bool = classifier::is_spam(&m1);
    println!("{}", score);
    println!("{}", is_spam);

    // Classifiy a typical ham message
    let m2 = String::from("Hi Bob, can you send me your machine learning homework?");
    let score: f32 = classifier::score(&m2);
    let is_spam: bool = classifier::is_spam(&m2);
    println!("{}", score);
    println!("{}", is_spam);
}
```

```bash
$> cargo run
0.99974066
true
0.0075160516
false
```

### Train your own model

```rust
extern crate bayespam;

use bayespam::classifier::Classifier;

fn main() {
    // Create a new classifier with an empty model
    let mut classifier = Classifier::new("my_super_model.json", true);

    // Train the model with a new spam example
    let spam = String::from("Don't forget our special promotion: -30% on men shoes, only today!");
    classifier.train(&spam, true);

    // Train the model with a new ham example
    let ham = String::from("Hi Bob, don't forget our meeting today at 4pm.");
    classifier.train(&ham, false);

    // Classify a typical spam message
    let m1 = String::from("Lose up to 19% weight. Special promotion on our new weightloss.");
    let score: f32 = classifier.score(&m1);
    let is_spam: bool = classifier.is_spam(&m1);
    println!("{}", score);
    println!("{}", is_spam);

    // Classifiy a typical ham message
    let m2 = String::from("Hi Bob, can you send me your machine learning homework?");
    let score: f32 = classifier.score(&m2);
    let is_spam: bool = classifier.is_spam(&m2);
    println!("{}", score);
    println!("{}", is_spam);
}
```

```bash
$> cargo run
0.89681536
true
0.00059083913
false
```

### Save your model

```rust
classifier.save("my_super_model.json")
```

```bash
$> cat my_super_model.json
{"spam_count_total":9,"ham_count_total":6,"token_table":{"men":[0,1],"dont":[1,1],"shoes":[0,1],"today":[1,1],"promotion:":[0,1],"only":[0,1],"bob":[1,0],"meeting":[1,0],"forget":[1,1],"our":[1,1],"special":[0,1]}}
```

## Documentation

Learn more about Bayespam here: [https://docs.rs/bayespam](https://docs.rs/bayespam).

## Contribution

Contributions via issues or pull requests are appreciated.

## License

Bayespam is distributed under the terms of the [MIT License](LICENSE).
