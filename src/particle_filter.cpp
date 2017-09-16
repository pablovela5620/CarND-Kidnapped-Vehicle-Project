/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#define _USE_MATH_DEFINES

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

bool debug = false;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    //  Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;
    num_particles = 200;

    //Create a Gaussian distribution for x, y, and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    //initialize every particle based on a random point on the gaussian distribution
    //set all particle weights to one
    for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.id = i;
        if (debug == true) {
            particle.x = x;
            particle.y = y;
            particle.theta = theta;

        } else {
            particle.x = dist_x(gen);
            particle.y = dist_y(gen);
            particle.theta = dist_theta(gen);

        }

        particle.weight = 1;

        particles.push_back(particle);
        weights.push_back(1);

    }


    is_initialized = true;
    //debugging console message
//    cout << "Particle: " << "x=" << particles[0].x << " y=" << particles[0].y << " theta=" << particles[0].theta
//         << " weight=" << particles[0].weight << endl;
//    cout << endl;
//    cout << "Initialized" << endl;
//    cout << endl;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    //  Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    for (int i = 0; i < num_particles; ++i) {

        double new_x;
        double new_y;
        double new_theta;

        // predicting new x, y and theta measurements for each particle depending on the yaw_rate
        if (yaw_rate == 0) {
            new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            new_theta = particles[i].theta;

        } else {
            new_x = particles[i].x +
                    (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            new_y = particles[i].y +
                    (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            new_theta = particles[i].theta + yaw_rate * delta_t;

        }

        //Adding random Gaussian noise to ensure particles don't collapse to the same measurement
        normal_distribution<double> dist_x(new_x, std_pos[0]);
        normal_distribution<double> dist_y(new_y, std_pos[0]);
        normal_distribution<double> dist_theta(new_theta, std_pos[0]);

        //remove Gaussian noise for prediction debugging
        if (debug == true) {
            particles[i].x = new_x;
            particles[i].y = new_y;
            particles[i].theta = new_theta;
        } else {
            particles[i].x = dist_x(gen);
            particles[i].y = dist_y(gen);
            particles[i].theta = dist_theta(gen);
        }


    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    //  Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html


    //immediately return for prediction debugging
    if (debug == true) {
        return;

    } else {

        //Denominator of multivariate gaussian does not change, keep outside of for loops
        double mg_denominator = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        //loop through all particles
        for (int i = 0; i < num_particles; ++i) {
            double multi_gauss = 1;

            //loop through all observations
            for (int j = 0; j < observations.size(); ++j) {

                //convert observations from car coordinate system to map coordinate system
                double obs_map_x = particles[i].x + (cos(particles[i].theta) * observations[j].x) -
                                   (sin(particles[i].theta) * observations[j].y);
                double obs_map_y = particles[i].y + (sin(particles[i].theta) * observations[j].x) +
                                   (cos(particles[i].theta) * observations[j].y);

//                cout << "Obs("<<j<<") (x,y)"<<"(" << observations[j].x << "," << observations[j].y << ")";
//                cout << "--->TObs("<<j<<") (x,y)"<<"(" << obs_map_x << "," << obs_map_y << ")"<<endl;

                //find the closest landmark, make list of all landmarks in map
                vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
                vector<double> landmark_object_distance(landmarks.size());

                for (int k = 0; k < landmarks.size(); ++k) {

                    //find the distance between the particle and the observed landmark
                    double landmark_particle_distance = dist(landmarks[k].x_f, landmarks[k].y_f, particles[i].x,
                                                             particles[i].y);

                    if (landmark_particle_distance <= sensor_range) {

                        landmark_object_distance[k] = dist(landmarks[k].x_f, landmarks[k].y_f, obs_map_x, obs_map_y);
                    } else {
                        //large number to ensure that objects that are too far are considered closest
                        landmark_object_distance[k] = numeric_limits<double>::max();

                    }

                }


                //nearest landmark neighbor
                int nearest = distance(landmark_object_distance.begin(),
                                       min_element(landmark_object_distance.begin(), landmark_object_distance.end()));

                double mu_x = landmarks[nearest].x_f;
                double mu_y = landmarks[nearest].y_f;
//                cout << "TObs("<<j<<") (x,y)"<<"(" << obs_map_x << "," << obs_map_y << ")";
//                cout << ": Predicted (x,y)"<<"(" << mu_x << "," << mu_y << ")"<<endl;


                //calculate gaussian
                double gauss_x = obs_map_x - mu_x;
                double gauss_y = obs_map_y - mu_y;
                multi_gauss *= (1 / mg_denominator) * exp(-(pow(gauss_x, 2) / (2 * pow(std_landmark[0], 2)) +
                                                            pow(gauss_y, 2) / (2 * pow(std_landmark[1], 2))));

            }


            particles[i].weight = multi_gauss;
            weights[i] = particles[i].weight;

        }
    }

}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    if (debug == true) {
        return;
    } else {
        default_random_engine gen;

        discrete_distribution<int> distribution(weights.begin(), weights.end());

        vector<Particle> resampled_particles;

        for (int i = 0; i < num_particles; ++i) {
            resampled_particles.push_back(particles[distribution(gen)]);
        }
        //replacing all existing particles with resampled particles 
        particles = resampled_particles;
    }

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
