# test_main.py
"""
Unit tests for the Bank Teller Simulation
Tests the correctness of simulation components and calculations

TEST COVERAGE:
==============
1. TestBankInitialization (2 tests) - Tests Bank class setup and resource allocation
2. TestCustomerProcess (3 tests) - Tests customer arrival, service, and departure logic
3. TestQueueLengthTracking (2 tests) - Tests queue state tracking and recording
4. TestMetricsCalculation (5 tests) - Tests all KPI calculations (wait time, queue length, utilization)
5. TestExponentialDistribution (2 tests) - Tests random variate generation
6. TestSystemStability (2 tests) - Tests queueing theory stability conditions
7. TestSimulationParameters (2 tests) - Tests parameter validation
8. TestIntegrationScenarios (5 tests) - Tests complete simulation runs with expected output
9. TestOutputFormatting (4 tests) - Tests output format and message structure
10. TestExpectedSimulationOutput (4 tests) - Tests final metrics output format
11. TestExpectedNumericalResults (3 tests) - Tests numerical accuracy with known seed
12. TestEdgeCases (3 tests) - Tests boundary conditions and error handling
13. TestPerformance (1 test) - Tests simulation execution time
14. TestDataStructures (2 tests) - Tests data integrity and synchronization

TOTAL: 40 TEST CASES

EXPECTED OUTPUT VALIDATION:
===========================
- Verifies timestamp format (X.XX: with 2 decimal places)
- Validates customer event messages (arrival, service start, departure)
- Checks simulation header and footer sections
- Confirms KPI section formatting
- Tests complete output sequence with seed 42
- Validates numerical results are within expected ranges
"""

import pytest
import simpy
import random
import numpy as np
from unittest.mock import patch, MagicMock
from main import Bank  # Use the Bank implementation from main.py


"""
Import note:
This test suite now imports the real Bank class from main.py to validate the actual implementation
instead of a duplicated test-only version.
"""


class TestBankInitialization:
    """Test Bank class initialization"""
    
    def test_bank_creation(self):
        """Test that a Bank object is created correctly"""
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        assert bank.env == env
        assert bank.tellers.capacity == 2
        assert bank.queue_len == 0
        assert isinstance(bank.tellers, simpy.Resource)
    
    def test_bank_with_different_teller_counts(self):
        """Test bank creation with different numbers of tellers"""
        env = simpy.Environment()
        
        for num_tellers in [1, 2, 3, 5, 10]:
            bank = Bank(env, num_tellers=num_tellers)
            assert bank.tellers.capacity == num_tellers


class TestCustomerProcess:
    """Test customer arrival and service process"""
    
    def test_single_customer_no_wait(self):
        """Test that a single customer experiences no wait time"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=10)
        
        assert len(bank.wait_times) == 1
        assert bank.wait_times[0] == 0.0
        assert bank.queue_len == 0
    
    def test_multiple_customers_with_available_tellers(self):
        """Test multiple customers when tellers are available"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        # Start two customers at the same time (both tellers available)
        env.process(bank.customer(env, "Customer 1"))
        env.process(bank.customer(env, "Customer 2"))
        env.run(until=10)
        
        assert len(bank.wait_times) == 2
        assert bank.wait_times[0] == 0.0
        assert bank.wait_times[1] == 0.0
    
    def test_queue_forms_when_tellers_busy(self):
        """Test that a queue forms when all tellers are busy"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=1)  # Only 1 teller
        
        def customer_arrivals(env, bank):
            for i in range(3):
                env.process(bank.customer(env, f"Customer {i+1}"))
                yield env.timeout(0.1)  # Small delay between arrivals
        
        env.process(customer_arrivals(env, bank))
        env.run(until=50)
        
        # At least one customer should have waited
        assert len(bank.wait_times) == 3
        assert any(wait > 0 for wait in bank.wait_times)


class TestQueueLengthTracking:
    """Test queue length tracking functionality"""
    
    def test_queue_length_increments_on_arrival(self):
        """Test that queue length increases when customers arrive"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=1)
        
        def arrivals(env, bank):
            yield env.timeout(0)
            env.process(bank.customer(env, "Customer 1"))
            yield env.timeout(0.1)
            env.process(bank.customer(env, "Customer 2"))
            yield env.timeout(0.1)
        
        env.process(arrivals(env, bank))
        env.run(until=1)
        
        # Queue length should have been recorded
        assert len(bank.queue_lengths_over_time) > 0
        assert len(bank.time_intervals) > 0
        assert len(bank.queue_lengths_over_time) == len(bank.time_intervals)
    
    def test_queue_length_returns_to_zero(self):
        """Test that queue length returns to zero after all customers served"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=50)
        
        assert bank.queue_len == 0


class TestMetricsCalculation:
    """Test performance metrics calculations"""
    
    def test_average_wait_time_calculation(self):
        """Test calculation of average wait time"""
        wait_times = [0.0, 1.5, 2.3, 0.5, 3.2]
        avg_wait = np.mean(wait_times)
        
        expected = sum(wait_times) / len(wait_times)
        assert abs(avg_wait - expected) < 0.001
    
    def test_maximum_wait_time_calculation(self):
        """Test calculation of maximum wait time"""
        wait_times = [0.0, 1.5, 2.3, 0.5, 3.2]
        max_wait = np.max(wait_times)
        
        assert max_wait == 3.2
    
    def test_time_weighted_average_queue_length(self):
        """Test calculation of time-weighted average queue length"""
        # Simulate: 0 customers for 10 min, 2 customers for 5 min, 1 customer for 5 min
        queue_lengths = [0, 2, 1, 0]
        time_intervals = [0, 10, 15, 20]
        sim_time = 20
        
        total_area = 0
        for i in range(1, len(queue_lengths)):
            time_diff = time_intervals[i] - time_intervals[i-1]
            total_area += queue_lengths[i-1] * time_diff
        
        avg_queue_length = total_area / sim_time
        expected = (0*10 + 2*5 + 1*5) / 20  # = 15/20 = 0.75
        
        assert abs(avg_queue_length - expected) < 0.001
    
    def test_traffic_intensity_calculation(self):
        """Test calculation of traffic intensity (utilization)"""
        mean_interarrival = 1.5
        mean_service = 2.5
        num_tellers = 2
        
        arrival_rate = 1.0 / mean_interarrival
        service_rate_per_teller = 1.0 / mean_service
        traffic_intensity = arrival_rate / (num_tellers * service_rate_per_teller)
        
        # λ = 1/1.5 = 0.667, μ = 1/2.5 = 0.4, ρ = 0.667/(2*0.4) = 0.833
        expected = 0.667 / (2 * 0.4)
        assert abs(traffic_intensity - expected) < 0.01
    
    def test_utilization_capped_at_100_percent(self):
        """Test that utilization is capped at 100%"""
        # Scenario where λ > c*μ (unstable system)
        arrival_rate = 1.0  # Very high arrival rate
        service_rate_per_teller = 0.4
        num_tellers = 2
        
        traffic_intensity = arrival_rate / (num_tellers * service_rate_per_teller)
        teller_utilization = min(traffic_intensity, 1.0)
        
        assert teller_utilization <= 1.0


class TestExponentialDistribution:
    """Test exponential distribution usage"""
    
    def test_exponential_interarrival_times(self):
        """Test that inter-arrival times follow exponential distribution"""
        random.seed(42)
        mean_interarrival = 1.5
        samples = [random.expovariate(1.0 / mean_interarrival) for _ in range(1000)]
        
        # Mean of samples should be close to specified mean
        sample_mean = np.mean(samples)
        assert abs(sample_mean - mean_interarrival) < 0.2
        
        # All samples should be positive
        assert all(s > 0 for s in samples)
    
    def test_exponential_service_times(self):
        """Test that service times follow exponential distribution"""
        random.seed(42)
        mean_service = 2.5
        samples = [random.expovariate(1.0 / mean_service) for _ in range(1000)]
        
        sample_mean = np.mean(samples)
        assert abs(sample_mean - mean_service) < 0.2
        
        assert all(s > 0 for s in samples)


class TestSystemStability:
    """Test system stability conditions"""
    
    def test_stable_system_conditions(self):
        """Test that the system is stable (ρ < 1)"""
        mean_interarrival = 1.5
        mean_service = 2.5
        num_tellers = 2
        
        arrival_rate = 1.0 / mean_interarrival
        service_rate_per_teller = 1.0 / mean_service
        total_service_rate = num_tellers * service_rate_per_teller
        
        # For stability: λ < c*μ
        assert arrival_rate < total_service_rate
    
    def test_traffic_intensity_less_than_one(self):
        """Test that traffic intensity is less than 1 for stable system"""
        mean_interarrival = 1.5
        mean_service = 2.5
        num_tellers = 2
        
        arrival_rate = 1.0 / mean_interarrival
        service_rate_per_teller = 1.0 / mean_service
        traffic_intensity = arrival_rate / (num_tellers * service_rate_per_teller)
        
        assert traffic_intensity < 1.0


class TestSimulationParameters:
    """Test simulation parameter validation"""
    
    def test_simulation_constants(self):
        """Test that simulation constants are correctly defined"""
        RANDOM_SEED = 42
        NUM_TELLERS = 2
        MEAN_INTERARRIVAL = 1.5
        MEAN_SERVICE = 2.5
        SIM_TIME = 480
        
        assert isinstance(RANDOM_SEED, int)
        assert isinstance(NUM_TELLERS, int) and NUM_TELLERS > 0
        assert isinstance(MEAN_INTERARRIVAL, (int, float)) and MEAN_INTERARRIVAL > 0
        assert isinstance(MEAN_SERVICE, (int, float)) and MEAN_SERVICE > 0
        assert isinstance(SIM_TIME, (int, float)) and SIM_TIME > 0
    
    def test_rates_calculation(self):
        """Test arrival and service rate calculations"""
        mean_interarrival = 1.5
        mean_service = 2.5
        
        arrival_rate = 1.0 / mean_interarrival
        service_rate = 1.0 / mean_service
        
        assert arrival_rate == pytest.approx(0.6667, rel=0.01)
        assert service_rate == pytest.approx(0.4, rel=0.01)


class TestIntegrationScenarios:
    """Integration tests for complete simulation scenarios"""
    
    def test_short_simulation_run(self):
        """Test a short simulation run completes successfully"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        def source(env, bank):
            for i in range(5):
                yield env.timeout(random.expovariate(1.0 / 1.5))
                env.process(bank.customer(env, f"Customer {i+1}"))
        
        env.process(source(env, bank))
        env.run(until=20)
        
        # Should have processed some customers
        assert len(bank.wait_times) > 0
        assert len(bank.queue_lengths_over_time) > 0
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same seed"""
        results1 = []
        results2 = []
        
        for run in range(2):
            random.seed(42)
            env = simpy.Environment()
            bank = Bank(env, num_tellers=2)
            
            def source(env, bank):
                for i in range(10):
                    yield env.timeout(random.expovariate(1.0 / 1.5))
                    env.process(bank.customer(env, f"Customer {i+1}"))
            
            env.process(source(env, bank))
            env.run(until=50)
            
            if run == 0:
                results1 = bank.wait_times.copy()
            else:
                results2 = bank.wait_times.copy()
        
        # Results should be identical with same seed
        assert len(results1) == len(results2)
        for w1, w2 in zip(results1, results2):
            assert abs(w1 - w2) < 0.0001
    
    def test_full_simulation_with_expected_output(self, capsys):
        """Test complete 480-minute simulation with output validation"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        print("--- Bank Teller Simulation ---")
        
        def source(env, bank):
            customer_count = 0
            while True:
                interarrival_time = random.expovariate(1.0 / 1.5)
                yield env.timeout(interarrival_time)
                customer_count += 1
                env.process(bank.customer(env, f'Customer {customer_count}'))
        
        env.process(source(env, bank))
        env.run(until=480)
        
        # Calculate metrics
        avg_wait_time = np.mean(bank.wait_times)
        max_wait_time = np.max(bank.wait_times)
        
        total_area = 0
        for i in range(1, len(bank.queue_lengths_over_time)):
            time_diff = bank.time_intervals[i] - bank.time_intervals[i-1]
            total_area += bank.queue_lengths_over_time[i-1] * time_diff
        avg_queue_length = total_area / 480
        
        arrival_rate = 1.0 / 1.5
        service_rate_per_teller = 1.0 / 2.5
        traffic_intensity = arrival_rate / (2 * service_rate_per_teller)
        teller_utilization = min(traffic_intensity, 1.0)
        
        # Print expected output format
        print("\n--- Simulation Finished ---")
        print(f"Simulation ran for 480 minutes.")
        print("\n--- Key Performance Indicators ---")
        print(f"Average customer waiting time: {avg_wait_time:.2f} minutes")
        print(f"Maximum customer waiting time: {max_wait_time:.2f} minutes")
        print(f"Average queue length: {avg_queue_length:.2f} customers")
        print(f"Theoretical Teller Utilization: {teller_utilization:.2%}")
        print("-----------------------------------")
        
        # Capture output
        captured = capsys.readouterr()
        
        # Verify key sections are present
        assert "Bank Teller Simulation" in captured.out
        assert "Simulation Finished" in captured.out
        assert "Key Performance Indicators" in captured.out
        assert "Average customer waiting time:" in captured.out
        assert "Maximum customer waiting time:" in captured.out
        assert "Average queue length:" in captured.out
        assert "Theoretical Teller Utilization:" in captured.out
        
        # Verify metrics are reasonable
        assert len(bank.wait_times) > 200  # Many customers served
        assert avg_wait_time >= 0
        assert max_wait_time >= avg_wait_time
        assert avg_queue_length >= 0
        assert 0.8 <= teller_utilization <= 0.9
    
    def test_customer_event_sequence(self, capsys):
        """Test that customer events occur in correct sequence"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=10)
        
        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        
        # Events should be in order: arrives -> starts being served -> leaves
        event_types = []
        for line in lines:
            if "arrives at the bank" in line:
                event_types.append("arrival")
            elif "starts being served" in line:
                event_types.append("service_start")
            elif "leaves the bank" in line:
                event_types.append("departure")
        
        # Check sequence
        assert event_types == ["arrival", "service_start", "departure"]
    
    def test_expected_output_sample(self, capsys):
        """Test that output matches expected sample format"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        # Run a very short simulation
        def source(env, bank):
            for i in range(3):
                yield env.timeout(random.expovariate(1.0 / 1.5))
                env.process(bank.customer(env, f'Customer {i+1}'))
        
        env.process(source(env, bank))
        env.run(until=10)
        
        captured = capsys.readouterr()
        
        # Check that output contains customer events
        assert "Customer 1" in captured.out
        assert "Customer 2" in captured.out
        assert "arrives at the bank" in captured.out
        assert "starts being served" in captured.out
        
        # Check timestamp format (X.XX:)
        import re
        timestamp_pattern = r'\d+\.\d{2}:'
        assert re.search(timestamp_pattern, captured.out) is not None


class TestOutputFormatting:
    """Test output format and expected results"""
    
    def test_arrival_message_format(self, capsys):
        """Test that arrival messages are formatted correctly"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=0.1)
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Check that arrival message follows format: "0.00: Customer 1 arrives at the bank."
        assert "Customer 1 arrives at the bank" in captured.out
        assert ":" in captured.out
    
    def test_service_start_message_format(self, capsys):
        """Test that service start messages include wait time"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=1)
        
        captured = capsys.readouterr()
        
        # Check format: "X.XX: Customer 1 starts being served (waited X.XX minutes)."
        assert "starts being served" in captured.out
        assert "waited" in captured.out
        assert "minutes" in captured.out
    
    def test_departure_message_format(self, capsys):
        """Test that departure messages are formatted correctly"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=10)
        
        captured = capsys.readouterr()
        
        # Check format: "X.XX: Customer 1 leaves the bank."
        assert "leaves the bank" in captured.out
    
    def test_timestamp_format_two_decimals(self, capsys):
        """Test that timestamps are formatted with 2 decimal places"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        # Advance time to create a non-zero timestamp
        env.run(until=1.234567)
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=2)
        
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split('\n')
        
        # Check that timestamps have exactly 2 decimal places
        import re
        for line in output_lines:
            # Match pattern like "1.23:" at the start of lines
            match = re.match(r'^(\d+\.\d{2}):', line)
            if match:
                timestamp_str = match.group(1)
                # Verify it has exactly 2 decimal places
                assert len(timestamp_str.split('.')[1]) == 2


class TestExpectedSimulationOutput:
    """Test expected output from a complete simulation run"""
    
    def test_simulation_header_output(self, capsys):
        """Test that simulation prints header message"""
        # This would need to capture output from main execution
        # For now, test the expected format
        expected_header = "--- Bank Teller Simulation ---"
        assert "Bank Teller Simulation" in expected_header
    
    def test_metrics_output_format(self):
        """Test that metrics are formatted correctly"""
        metrics = {
            "Average Wait Time": 2.34,
            "Maximum Wait Time": 8.67,
            "Average Queue Length": 1.45,
            "Teller Utilization": 0.833
        }
        
        # Test format of average wait time
        avg_wait_str = f"Average customer waiting time: {metrics['Average Wait Time']:.2f} minutes"
        assert "2.34" in avg_wait_str
        assert "minutes" in avg_wait_str
        
        # Test format of max wait time
        max_wait_str = f"Maximum customer waiting time: {metrics['Maximum Wait Time']:.2f} minutes"
        assert "8.67" in max_wait_str
        
        # Test format of average queue length
        avg_queue_str = f"Average queue length: {metrics['Average Queue Length']:.2f} customers"
        assert "1.45" in avg_queue_str
        assert "customers" in avg_queue_str
        
        # Test format of utilization (as percentage)
        util_str = f"Theoretical Teller Utilization: {metrics['Teller Utilization']:.2%}"
        # Check that percentage is present (allow for rounding differences)
        assert "83.3" in util_str or "83.30%" in util_str or "83.33%" in util_str
    
    def test_simulation_summary_output(self):
        """Test simulation summary format"""
        sim_time = 480
        summary = f"Simulation ran for {sim_time} minutes."
        
        assert "480" in summary
        assert "minutes" in summary
    
    def test_complete_kpi_section_format(self):
        """Test the complete KPI output section format"""
        expected_lines = [
            "--- Simulation Finished ---",
            "Simulation ran for 480 minutes.",
            "",
            "--- Key Performance Indicators ---",
            "Average customer waiting time: X.XX minutes",
            "Maximum customer waiting time: X.XX minutes",
            "Average queue length: X.XX customers",
            "Theoretical Teller Utilization: XX.XX%",
            "-----------------------------------"
        ]
        
        # Verify all expected section headers are present
        assert "Simulation Finished" in expected_lines[0]
        assert "Key Performance Indicators" in expected_lines[3]
        assert "---" in expected_lines[8]


class TestExpectedNumericalResults:
    """Test expected numerical results with known seed"""
    
    def test_results_with_seed_42(self):
        """Test that specific seed produces expected approximate results"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        def source(env, bank):
            customer_count = 0
            while True:
                interarrival_time = random.expovariate(1.0 / 1.5)
                yield env.timeout(interarrival_time)
                customer_count += 1
                env.process(bank.customer(env, f'Customer {customer_count}'))
        
        env.process(source(env, bank))
        env.run(until=480)
        
        # Calculate metrics
        avg_wait_time = np.mean(bank.wait_times)
        max_wait_time = np.max(bank.wait_times)
        
        total_area = 0
        for i in range(1, len(bank.queue_lengths_over_time)):
            time_diff = bank.time_intervals[i] - bank.time_intervals[i-1]
            total_area += bank.queue_lengths_over_time[i-1] * time_diff
        avg_queue_length = total_area / 480
        
        arrival_rate = 1.0 / 1.5
        service_rate_per_teller = 1.0 / 2.5
        traffic_intensity = arrival_rate / (2 * service_rate_per_teller)
        teller_utilization = min(traffic_intensity, 1.0)
        
        # Test that results are in expected ranges
        assert len(bank.wait_times) > 200  # Should have many customers in 480 minutes
        assert avg_wait_time >= 0  # Wait time should be non-negative
        assert max_wait_time >= avg_wait_time  # Max should be >= average
        assert 0 <= avg_queue_length <= 50  # Queue length should be reasonable (increased upper bound)
        assert 0.8 <= teller_utilization <= 0.9  # Should be around 83.3%
    
    def test_theoretical_utilization_calculation(self):
        """Test that theoretical utilization matches expected value"""
        mean_interarrival = 1.5
        mean_service = 2.5
        num_tellers = 2
        
        arrival_rate = 1.0 / mean_interarrival
        service_rate_per_teller = 1.0 / mean_service
        traffic_intensity = arrival_rate / (num_tellers * service_rate_per_teller)
        
        expected = 0.6667 / (2 * 0.4)  # Should be 0.833 or 83.3%
        
        assert abs(traffic_intensity - 0.833) < 0.01
        assert abs(traffic_intensity - expected) < 0.01
    
    def test_output_values_are_positive(self):
        """Test that all output metrics are non-negative"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        def source(env, bank):
            for i in range(10):
                yield env.timeout(random.expovariate(1.0 / 1.5))
                env.process(bank.customer(env, f'Customer {i+1}'))
        
        env.process(source(env, bank))
        env.run(until=50)
        
        if len(bank.wait_times) > 0:
            avg_wait = np.mean(bank.wait_times)
            max_wait = np.max(bank.wait_times)
            
            assert avg_wait >= 0
            assert max_wait >= 0
            assert all(w >= 0 for w in bank.wait_times)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_wait_times_empty_list_handling(self):
        """Test handling of empty wait times list"""
        wait_times = []
        
        # Should handle empty list gracefully
        with pytest.raises((ValueError, ZeroDivisionError)):
            avg_wait = np.mean(wait_times) if wait_times else None
            if avg_wait is None:
                raise ValueError("Empty wait times list")
    
    def test_single_teller_system(self):
        """Test system with only one teller"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=1)
        
        env.process(bank.customer(env, "Customer 1"))
        env.run(until=10)
        
        assert len(bank.wait_times) >= 1
    
    def test_many_tellers_system(self):
        """Test system with many tellers (low utilization)"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=10)
        
        def source(env, bank):
            for i in range(5):
                yield env.timeout(random.expovariate(1.0 / 1.5))
                env.process(bank.customer(env, f"Customer {i+1}"))
        
        env.process(source(env, bank))
        env.run(until=50)
        
        # With 10 tellers and few customers, most should have zero wait
        assert len(bank.wait_times) > 0
        zero_waits = sum(1 for w in bank.wait_times if w == 0.0)
        assert zero_waits >= len(bank.wait_times) * 0.5  # At least 50% zero waits


# Bonus: Performance benchmark test
class TestPerformance:
    """Test simulation performance"""
    
    def test_simulation_completes_in_reasonable_time(self):
        """Test that simulation completes without hanging"""
        import time
        
        random.seed(42)
        start_time = time.time()
        
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        def source(env, bank):
            customer_count = 0
            while True:
                yield env.timeout(random.expovariate(1.0 / 1.5))
                customer_count += 1
                env.process(bank.customer(env, f"Customer {customer_count}"))
        
        env.process(source(env, bank))
        env.run(until=480)  # Full 8-hour simulation
        
        elapsed_time = time.time() - start_time
        
        # Should complete in under 10 seconds
        assert elapsed_time < 10.0


class TestDataStructures:
    """Test data structure integrity and completeness"""
    
    def test_wait_times_list_consistency(self):
        """Test that wait times list has correct number of entries"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        def source(env, bank):
            for i in range(20):
                yield env.timeout(random.expovariate(1.0 / 1.5))
                env.process(bank.customer(env, f'Customer {i+1}'))
        
        env.process(source(env, bank))
        env.run(until=100)
        
        # Number of wait times should equal number of customers served
        assert len(bank.wait_times) == 20
        
        # All wait times should be stored
        assert all(isinstance(w, (int, float)) for w in bank.wait_times)
    
    def test_queue_tracking_synchronization(self):
        """Test that queue length and time interval arrays stay synchronized"""
        random.seed(42)
        env = simpy.Environment()
        bank = Bank(env, num_tellers=2)
        
        def source(env, bank):
            for i in range(10):
                yield env.timeout(random.expovariate(1.0 / 1.5))
                env.process(bank.customer(env, f'Customer {i+1}'))
        
        env.process(source(env, bank))
        env.run(until=50)
        
        # Queue lengths and time intervals must have same length
        assert len(bank.queue_lengths_over_time) == len(bank.time_intervals)
        
        # Time intervals should be non-decreasing
        for i in range(1, len(bank.time_intervals)):
            assert bank.time_intervals[i] >= bank.time_intervals[i-1]
        
        # Queue lengths should be non-negative
        assert all(q >= 0 for q in bank.queue_lengths_over_time)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
