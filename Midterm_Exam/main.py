# MIDTERM EXAM
# CSEC 413 - MODELING AND SIMULATION
# Herald Carl N. Avila
# BSCS - 4B

import simpy
import random
import numpy as np

# ============================================================================
# SIMULATION CONSTANTS
# ============================================================================
RANDOM_SEED = 42
NUM_TELLERS = 2
MEAN_INTERARRIVAL = 1.5  # minutes
MEAN_SERVICE = 2.5       # minutes
SIM_TIME = 480           # minutes (8 hours)


# ============================================================================
# BANK CLASS
# ============================================================================
class Bank:
    """
    A bank simulation with multiple tellers using SimPy.
    
    Attributes:
        env (simpy.Environment): The simulation environment
        tellers (simpy.Resource): Pool of teller resources
        mean_interarrival (float): Mean time between customer arrivals
        mean_service (float): Mean service time per customer
        wait_times (list): Recorded wait times for all customers
        queue_lengths_over_time (list): Queue lengths at each change
        time_intervals (list): Timestamps of queue length changes
        queue_len (int): Current queue length
        arrivals (int): Total number of customer arrivals
    """
    
    def __init__(self, env, num_tellers, mean_interarrival=MEAN_INTERARRIVAL, 
                 mean_service=MEAN_SERVICE):
        """
        Initialize the Bank simulation.
        
        Args:
            env (simpy.Environment): SimPy environment
            num_tellers (int): Number of teller counters
            mean_interarrival (float): Mean inter-arrival time
            mean_service (float): Mean service time
        """
        self.env = env
        self.tellers = simpy.Resource(env, num_tellers)
        self.mean_interarrival = mean_interarrival
        self.mean_service = mean_service
        
        # Initialize performance tracking metrics
        self.wait_times = []
        self.queue_lengths_over_time = []
        self.time_intervals = []
        self.queue_len = 0
        self.arrivals = 0

    def customer(self, env, name):
        """
        Simulate a customer's journey through the bank.
        
        Process:
        1. Customer arrives and joins the queue
        2. Customer waits for an available teller
        3. Customer gets served
        4. Customer leaves the bank
        
        Args:
            env (simpy.Environment): SimPy environment
            name (str): Customer identifier
        """
        arrival_time = env.now
        self._log_arrival(name, arrival_time)
        
        with self.tellers.request() as request:
            # Track queue length at arrival
            current_queue_length = len(self.tellers.queue)
            self._record_queue_state(current_queue_length)
            
            # Wait for teller availability
            yield request
            
            # Calculate and record wait time
            wait_time = env.now - arrival_time
            self.wait_times.append(wait_time)
            self._log_service_start(name, wait_time)
            
            # Perform service
            service_duration = self._generate_service_time()
            yield env.timeout(service_duration)
            
            self._log_departure(name)

    def arrival_process(self):
        """
        Generate customer arrivals at exponentially distributed intervals.
        
        This process runs continuously throughout the simulation, creating
        new customer processes at random intervals following an exponential
        distribution.
        """
        customer_id = 0
        
        while True:
            # Generate inter-arrival time
            interarrival_time = self._generate_interarrival_time()
            yield self.env.timeout(interarrival_time)
            
            # Create new customer
            customer_id += 1
            self.arrivals += 1
            self.env.process(self.customer(self.env, f"Customer {customer_id}"))

    def report(self, simulation_time):
        """
        Calculate and display simulation results.
        
        Computes and prints:
        - Average customer wait time
        - Maximum customer wait time
        - Time-weighted average queue length
        - Theoretical teller utilization
        
        Args:
            simulation_time (float): Total simulation duration
        """
        # Calculate wait time metrics
        avg_wait_time = self._calculate_average_wait()
        max_wait_time = self._calculate_max_wait()
        
        # Calculate time-weighted average queue length
        avg_queue_length = self._calculate_average_queue_length(simulation_time)
        
        # Calculate theoretical utilization
        teller_utilization = self._calculate_utilization()
        
        # Display results
        self._print_report(simulation_time, avg_wait_time, max_wait_time, 
                          avg_queue_length, teller_utilization)

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _generate_interarrival_time(self):
        """Generate exponentially distributed inter-arrival time."""
        return random.expovariate(1.0 / self.mean_interarrival)
    
    def _generate_service_time(self):
        """Generate exponentially distributed service time."""
        return random.expovariate(1.0 / self.mean_service)
    
    def _record_queue_state(self, queue_length):
        """
        Record the current queue state for time-weighted averaging.
        
        Args:
            queue_length (int): Current number of customers in queue
        """
        self.queue_lengths_over_time.append(queue_length)
        self.time_intervals.append(self.env.now)
    
    def _calculate_average_wait(self):
        """Calculate average customer wait time."""
        return np.mean(self.wait_times) if self.wait_times else 0.0
    
    def _calculate_max_wait(self):
        """Calculate maximum customer wait time."""
        return np.max(self.wait_times) if self.wait_times else 0.0
    
    def _calculate_average_queue_length(self, simulation_time):
        """
        Calculate time-weighted average queue length.
        
        Uses the trapezoidal rule to compute the area under the queue
        length curve and divides by total simulation time.
        
        Args:
            simulation_time (float): Total simulation duration
            
        Returns:
            float: Time-weighted average queue length
        """
        if not self.queue_lengths_over_time or simulation_time <= 0:
            return 0.0
        
        total_area = 0.0
        for i in range(1, len(self.queue_lengths_over_time)):
            time_difference = self.time_intervals[i] - self.time_intervals[i - 1]
            queue_level = self.queue_lengths_over_time[i - 1]
            total_area += queue_level * time_difference
        
        return total_area / simulation_time
    
    def _calculate_utilization(self):
        """
        Calculate theoretical teller utilization (traffic intensity).
        
        Uses queueing theory formula: ρ = λ / (c * μ)
        where λ is arrival rate, μ is service rate, c is number of servers
        
        Returns:
            float: Utilization factor (capped at 1.0)
        """
        arrival_rate = 1.0 / self.mean_interarrival
        service_rate = 1.0 / self.mean_service
        num_tellers = self.tellers.capacity
        
        traffic_intensity = arrival_rate / (num_tellers * service_rate)
        return min(traffic_intensity, 1.0)
    
    # ========================================================================
    # LOGGING METHODS
    # ========================================================================
    
    def _log_arrival(self, customer_name, time):
        """Log customer arrival event."""
        print(f"{time:.2f}: {customer_name} arrives at the bank.")
    
    def _log_service_start(self, customer_name, wait_time):
        """Log service start event."""
        print(f"{self.env.now:.2f}: {customer_name} starts being served "
              f"(waited {wait_time:.2f} minutes).")
    
    def _log_departure(self, customer_name):
        """Log customer departure event."""
        print(f"{self.env.now:.2f}: {customer_name} leaves the bank.")
    
    def _print_report(self, sim_time, avg_wait, max_wait, avg_queue, utilization):
        """
        Print formatted simulation report.
        
        Args:
            sim_time (float): Simulation duration
            avg_wait (float): Average wait time
            max_wait (float): Maximum wait time
            avg_queue (float): Average queue length
            utilization (float): Teller utilization rate
        """
        print("\n--- Simulation Finished ---")
        print(f"Simulation ran for {sim_time:.0f} minutes.\n")
        print("--- Key Performance Indicators ---")
        print(f"Average customer waiting time: {avg_wait:.2f} minutes")
        print(f"Maximum customer waiting time: {max_wait:.2f} minutes")
        print(f"Average queue length: {avg_queue:.2f} customers")
        print(f"Theoretical Teller Utilization: {utilization:.2%}")


# ============================================================================
# SIMULATION EXECUTION
# ============================================================================
def run_simulation(random_seed=RANDOM_SEED, num_tellers=NUM_TELLERS,
                   mean_interarrival=MEAN_INTERARRIVAL, 
                   mean_service=MEAN_SERVICE, sim_time=SIM_TIME):
    """
    Execute the bank teller simulation.
    
    Args:
        random_seed (int): Seed for reproducibility
        num_tellers (int): Number of teller counters
        mean_interarrival (float): Mean customer inter-arrival time
        mean_service (float): Mean service time
        sim_time (float): Total simulation time
        
    Returns:
        Bank: The Bank object with simulation results
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create simulation environment and bank
    env = simpy.Environment()
    bank = Bank(env, num_tellers, mean_interarrival, mean_service)
    
    # Start customer arrival process
    env.process(bank.arrival_process())
    
    # Run simulation
    env.run(until=sim_time)
    
    # Generate report
    bank.report(sim_time)
    
    return bank


def main():
    """Main entry point for the simulation."""
    run_simulation()


if __name__ == "__main__":
    main()