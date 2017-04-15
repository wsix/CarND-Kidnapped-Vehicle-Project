/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    std::default_random_engine gen;
    num_particles = 200;
    
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_yaw(theta, std[2]);
    
    for (int i = 0; i < num_particles; ++i) {
        double particle_x = dist_x(gen);
        double particle_y = dist_y(gen);
        double particle_yaw = dist_yaw(gen);
        Particle temp = {i, particle_x, particle_y, particle_yaw, 1.0};
        particles.push_back(temp);
        weights.push_back(1.0f); 
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0.0, std_pos[0]);    
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_yaw(0.0, std_pos[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle& particle = particles[i];
        double particle_x = particle.x;
        double particle_y = particle.y;
        double theta = particle.theta;

        particle.theta = theta + yaw_rate * delta_t;

        if (yaw_rate > 0.00001) {
            double radius = velocity / yaw_rate;
            particle_x += radius * ( sin(particle.theta) - sin(theta) );
            particle_y += radius * ( cos(theta) - cos(particle.theta) );
        } else {
            particle_x += velocity * delta_t * cos(theta);
            particle_y += velocity * delta_t * sin(theta);
        }

        particle.x = particle_x + dist_x(gen);
        particle.y = particle_y + dist_y(gen);
        particle.theta += dist_yaw(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     Map map_landmarks,
                                     std::vector<LandmarkObs>& associations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < predicted.size(); ++i) {
        LandmarkObs association;
        double min_distance_square;
        
        association.id = map_landmarks.landmark_list[0].id_i;
        association.x = map_landmarks.landmark_list[0].x_f;
        association.y = map_landmarks.landmark_list[0].y_f;
        min_distance_square = pow((map_landmarks.landmark_list[0].x_f - predicted[i].x), 2) + 
            pow((map_landmarks.landmark_list[0].y_f - predicted[i].y), 2);
        
        for (int j = 1; j < map_landmarks.landmark_list.size(); ++j) {
            double distance_square = pow((map_landmarks.landmark_list[j].x_f - predicted[i].x), 2) + 
                pow((map_landmarks.landmark_list[j].y_f - predicted[i].y), 2);
            if (distance_square < min_distance_square) {
                min_distance_square = distance_square;
                association.id = map_landmarks.landmark_list[j].id_i;
                association.x = map_landmarks.landmark_list[j].x_f;
                association.y = map_landmarks.landmark_list[j].y_f;
            }
        }

        associations.push_back(association);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
    for (int i = 0; i < num_particles; ++i) {
        Particle& particle = particles[i];
        std::vector<LandmarkObs> predicted;
        std::vector<LandmarkObs> associations;
        double weight = 1.0;

        // Transformation step.
        for (int j = 0; j < observations.size(); ++j) {
            LandmarkObs landmark_pred;
            landmark_pred.id = observations[j].id;
            landmark_pred.x = observations[j].x * cos(particle.theta) - 
                observations[j].y * sin(particle.theta) + particle.x;
            landmark_pred.y = observations[j].x * sin(particle.theta) +
                observations[j].y * cos(particle.theta) + particle.y;
            predicted.push_back(landmark_pred);
        }

        // Associations step.
        dataAssociation(predicted, map_landmarks, associations);

        // Calculating particle's final weight.
        for (int j = 0; j < predicted.size(); ++j) {
            LandmarkObs predict = predicted[j];
            LandmarkObs association = associations[j];
            double delta_x = predict.x - association.x;
            double delta_y = predict.y - association.y;
            
            double p_x_y = 1 / (2 * M_PI * std_landmark[0] * std_landmark[0]) *
                exp(-(delta_x * delta_x / (2 * std_landmark[0] * std_landmark[0]) +
                      delta_y * delta_y / (2 * std_landmark[0] * std_landmark[0])));
            weight *= p_x_y;
        }
        particle.weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::default_random_engine gen;
    std::discrete_distribution<> dd(weights.begin(), weights.end());
    std::vector<Particle> newParticles;
    for (int i = 0; i < num_particles; ++i)
        newParticles.push_back(particles[dd(gen)]);

    particles = newParticles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
