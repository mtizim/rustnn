# Running the code

Actually compiling the code requires running `cargo run`

`cargo` can be acquired by installing `rustup`.
https://doc.rust-lang.org/cargo/getting-started/installation.html

With a seeded initialization everything is deterministic, and the training for the target MSE takes a bit long, so I'm also pasting the output of the program under the `main` function (in `main.rs`)

The relevant momentum and RMSprop implementations are in `optimizers/`
