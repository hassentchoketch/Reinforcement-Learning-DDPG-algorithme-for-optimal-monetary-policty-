# Replication of the paper "Optimal Monetary Policy Using Reinforcement Learning"
This repository contains the Python code for replicating the research paper: 
Natascha Hinterlang,Natascha Hinterlang(2021) "Optimal monetary policyusing reinforcement learning",Discussion Paper,Deutsche Bundesbank No 51

## Abstract
This paper introduces a reinforcement learning based approach to compute optimal interest rate reaction functions in terms of fulfilling inflation and output gap targets. Themethod is generally flexible enough to incorporate restrictions like the zero lower bound, nonlinear economy structures or asymmetric preferences. We use quarterly U.S. data from 1987:Q3-2007:Q2 to estimate (nonlinear) model transition equations, train optimal policies and perform counterfactual analyses to evaluate them, assuming that the transition equations remain unchanged. All of our resulting policy rules outperform other common rules as well as the actual federal funds rate. Given a neural network representation of
the economy, our optimized nonlinear policy rules reduce the central bank’s loss by over 43 %. A DSGE model comparison exercise further indicates robustness of the optimized rules

## Current Status
This project is currently a work in progress and is not yet complete. We are still in the process of replicating the results presented in the paper and fine-tuning our implementation.

Future Work
Once our implementation is complete, we plan to evaluate the performance of our approach on additional datasets and compare it to traditional models of monetary policy. We also hope to explore extensions and variations of the approach presented in the paper.
