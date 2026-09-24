import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize


# ==========================================
# 1. QUANTITATIVE OPTIMIZATION ENGINE
# ==========================================

def roys_safety_first_objective(weights, mean_returns, cov_matrix, hurdle_rate):
    """Minimizes the negative Roy's Safety-First Ratio."""
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    if port_vol == 0:
        return 1e6
        
    sfr = (port_return - hurdle_rate) / port_vol
    return -sfr

def optimize_annual_sfr(mean_returns, cov_matrix, hurdle_rate):
    """Solves for optimal weights maximizing SFR for a given hurdle."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, hurdle_rate)
    init_guess = np.array([1.0 / num_assets] * num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
    result = minimize(
        roys_safety_first_objective,
        init_guess,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return {
        'optimal_weights': result.x,
        'expected_return': np.dot(result.x, mean_returns),
        'expected_volatility': np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))),
        'max_sfr': -result.fun
    }

# ==========================================
# 2. MONTE CARLO STRESS TEST ENGINE
# ==========================================

def run_monte_carlo_sim(weights, mean_returns, cov_matrix, initial_value, annual_withdrawal, years, num_simulations=5000):
    """Vectorized Monte Carlo simulation factoring in dynamic cash outflows."""
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    paths = np.zeros((years + 1, num_simulations))
    paths[0] = initial_value
    random_shocks = np.random.normal(0, 1, size=(years, num_simulations))
    
    for t in range(1, years + 1):
        drift = port_return - (0.5 * port_vol ** 2)
        shock = port_vol * random_shocks[t-1]
        
        # Grow portfolio, then subtract cash flows (assuming end-of-year withdrawal)
        end_of_year_value = paths[t-1] * np.exp(drift + shock)
        paths[t] = np.maximum(end_of_year_value - annual_withdrawal, 0)
        
    return paths

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================

def main():
    st.set_page_config(page_title="SFR Optimization Engine", layout="wide")
    st.title("Asset-Liability Optimization Dashboard")
    
    # --- Capital Market Assumptions (Mock Data) ---
    assets = ["Global Equities", "Core Fixed Income", "Return Stacked Alts"]
    expected_returns = np.array([0.08, 0.045, 0.065])
    cov_matrix = np.array([
        [0.0256, 0.0010, 0.0120],
        [0.0010, 0.0036, 0.0015],
        [0.0120, 0.0015, 0.0144]
    ])

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("Client Parameters")
        initial_wealth = st.number_input("Initial Portfolio Value ($)", value=2500000, step=100000)
        annual_withdrawal = st.number_input("Target Annual Withdrawal ($)", value=125000, step=5000)
        time_horizon = st.slider("Plan Horizon (Years)", 5, 40, 20)
        
        st.markdown("---")
        st.header("Simulation Settings")
        num_sims = st.slider("Paths", 1000, 10000, 5000, step=1000)
        
        # Calculate Required Hurdle Rate natively
        implied_hurdle = annual_withdrawal / initial_wealth
        st.metric("Implied Target Hurdle Rate", f"{implied_hurdle:.2%}")

    # --- Optimization Execution ---
    opt_results = optimize_annual_sfr(expected_returns, cov_matrix, implied_hurdle)
    opt_weights = opt_results['optimal_weights']
    
    # --- Dashboard Layout ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimized Expected Return", f"{opt_results['expected_return']:.2%}")
    col2.metric("Portfolio Volatility", f"{opt_results['expected_volatility']:.2%}")
    col3.metric("Maximized SFR Score", f"{opt_results['max_sfr']:.2f}")
    
    st.markdown("### Target Glide Path Allocation")
    weights_df = pd.DataFrame([opt_weights], columns=assets).style.format("{:.1%}")
    st.dataframe(weights_df, use_container_width=True)
    
    st.markdown("---")
    
    # --- Monte Carlo Execution ---
    st.markdown("### 📊 Sequence of Returns Stress Test")
    
    if st.button("Run Monte Carlo Stress Test", type="primary"):
        with st.spinner(f"Simulating {num_sims} market environments..."):
            
            paths = run_monte_carlo_sim(
                opt_weights, expected_returns, cov_matrix, 
                initial_wealth, annual_withdrawal, 
                years=time_horizon, num_simulations=num_sims
            )
            
            # Extract Percentiles
            percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
            x_axis = list(range(time_horizon + 1))
            
            # Plotly Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=percentiles[2], mode='lines', name='Median Outcome', line=dict(color='#2ca02c', width=3)))
            fig.add_trace(go.Scatter(x=x_axis, y=percentiles[3], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x_axis, y=percentiles[1], mode='lines', fill='tonexty', name='Middle 50%', fillcolor='rgba(44, 160, 44, 0.3)', line=dict(width=0)))
            fig.add_trace(go.Scatter(x=x_axis, y=percentiles[4], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x_axis, y=percentiles[0], mode='lines', fill='tonexty', name='90% Confidence', fillcolor='rgba(44, 160, 44, 0.1)', line=dict(width=0)))
            
            # Formatting
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Depletion")
            fig.update_layout(height=500, hovermode="x unified", yaxis_tickformat="$,.0f", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Success Probability Metric
            final_values = paths[-1]
            success_rate = np.sum(final_values > 0) / num_sims
            st.success(f"**Plan Success Probability (Avoids Depletion): {success_rate:.1%}**")

if __name__ == "__main__":
    main()
