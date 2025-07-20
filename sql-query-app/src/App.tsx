import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import './App.css'; // Import the main CSS file
import Typewriter from './Typewriter';
import History from './History'; // Import the new History component

// Import Redux actions and types
import { RootState, AppDispatch } from './store/store';
import {
  setInputQuestion,
  resetChartState,
  generateSqlQuery,
  generateChart,
  setSqlContentWriterDone,
  setChartLabelWriterDone,
  loadAppStateFromEncodedUrl,
} from './store/chartSlice';

// Define the ChartApiResponse type locally for clarity, matching the slice
type ChartApiResponse = {
  sql_query_gen_by_model: string;
  chart_id: string;
  chart_url: string;
  user_id: string;
  chart_type: string;
  timestamp: number;
  question: string;
};

// Main App component
const App: React.FC = () => {
  // Use useSelector to access state from the Redux store
  const {
    inputQuestion,
    sqlQuery,
    chartData,
    loadingSql,
    loadingChart,
    error, // This is the state that holds the error message
    sqlContentWriterDone,
    chartLabelWriterDone,
  } = useSelector((state: RootState) => state.chart); // Access the 'chart' slice

  // Use useDispatch to dispatch actions
  const dispatch: AppDispatch = useDispatch();

  // Effect to load state from URL on initial mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const encodedChartData = params.get('chartData');

    if (encodedChartData) {
      try {
        // Decode Base64 and parse JSON
        const decodedString = atob(encodedChartData);
        const parsedChartData: ChartApiResponse = JSON.parse(decodedString);
        // Dispatch action to load this data into Redux state
        dispatch(loadAppStateFromEncodedUrl(parsedChartData));
      } catch (e) {
        console.error("Failed to parse chart data from URL:", e);
        // Optionally, dispatch an error action or reset state
        dispatch(resetChartState());
      }
    }
  }, [dispatch]); // Only run once on mount

  // Effect to update URL when chartData changes
  useEffect(() => {
    if (chartData && chartLabelWriterDone) { // Only update URL when chart is fully displayed
      try {
        const stringifiedData = JSON.stringify(chartData);
        const encodedData = btoa(stringifiedData); // Base64 encode
        const newUrl = `${window.location.pathname}?chartData=${encodedData}`;
        window.history.replaceState({}, '', newUrl); // Update URL without reloading
      } catch (e) {
        console.error("Failed to encode chart data for URL:", e);
      }
    } else if (!chartData && window.location.search.includes('chartData')) {
      // If chartData is cleared but URL still has it, clear the URL param
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, [chartData, chartLabelWriterDone]); // Re-run when chartData or its label typewriter status changes


  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // Only proceed if there's an input question and not already loading
    if (!inputQuestion.trim() || loadingSql || loadingChart) return;

    dispatch(resetChartState()); // Reset main display state before new submission
    // Dispatch the async thunk to generate SQL
    dispatch(generateSqlQuery(inputQuestion));
  };

  // Effect hook to trigger chart generation after SQL query is available and its content is typed
  useEffect(() => {
    // Only proceed if SQL query is available, its content has finished typing,
    // chart is not loading, chartData is null, AND there is no active error.
    if (sqlQuery && sqlContentWriterDone && !loadingChart && !chartData && !error) {
      // Dispatch the async thunk to generate the chart, passing the original question
      dispatch(generateChart({ sqlQuery, question: inputQuestion }));
    }
    // Dependency array: re-run this effect when sqlQuery, sqlContentWriterDone, loadingChart, chartData, or error changes
  }, [sqlQuery, sqlContentWriterDone, dispatch, loadingChart, chartData, inputQuestion, error]);

  return (
    <div className="app-container">
      {/* History Component */}
      <History />

      {/* Main Content Area */}
      <div className="main-content-area">
        <h2 className="app-title">SQL Query and Chart Generator</h2>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            placeholder="Enter your question (e.g., 'Show me sales data')"
            value={inputQuestion} // Use Redux state for input value
            onChange={e => dispatch(setInputQuestion(e.target.value))} // Dispatch action on change
            className="input-field"
          />
          <button
            type="submit"
            disabled={loadingSql || loadingChart || !inputQuestion.trim()}
            className="submit-button"
          >
            {loadingSql || loadingChart ? 'Processing...' : 'Submit'}
          </button>
        </form>

        {/* SQL Query Generation Status (Typewriter) */}
        {loadingSql && (
          <p className="status-message">
            <Typewriter text="Generating SQL query..." className="typewriter-message" />
          </p>
        )}

        {/* Display SQL Query Label (Static) and Content (Typewriter) */}
        {sqlQuery && !loadingSql && (
          <div className="sql-display-section">
            {/* Static label */}
            <p className="sql-label">The generated SQL query is:</p>

            {/* SQL content with typewriter effect */}
            {!sqlContentWriterDone ? (
              <pre className="sql-box">
                <Typewriter
                  text={sqlQuery}
                  className="typewriter-content"
                  onDone={() => dispatch(setSqlContentWriterDone(true))}
                  speed={30} // Slower speed for SQL query content
                />
              </pre>
            ) : (
              // Static SQL content once typewriter is done
              <pre className="sql-box">
                {sqlQuery}
              </pre>
            )}
          </div>
        )}

        {/* Chart Generation Status (Typewriter) */}
        {loadingChart && sqlQuery && sqlContentWriterDone && !chartData && ( // Depend on sqlContentWriterDone
          <p className="status-message">
            <Typewriter text="Generating Chart..." className="typewriter-message" />
          </p>
        )}

        {/* Display Chart with Typewriter effect on its label */}
        {chartData && (
          <div className="chart-display-section">
            {/* Typewriter for the chart title */}
            {!chartLabelWriterDone ? (
              <h3 className="chart-title-typewriter">
                <Typewriter
                  text={`Chart: ${chartData.chart_type.toUpperCase()}`}
                  className="typewriter-content"
                  onDone={() => dispatch(setChartLabelWriterDone(true))}
                  speed={50} // Adjust speed as needed
                />
              </h3>
            ) : (
              // Static title once typewriter is done
              <h3 className="chart-title">Chart: {chartData.chart_type.toUpperCase()}</h3>
            )}

            {/* Iframe only visible after chart label typewriter is done */}
            {chartLabelWriterDone && (
              <iframe
                src={chartData.chart_url}
                title={`Chart: ${chartData.chart_type}`}
                width="100%"
                height="400"
                frameBorder="0"
                allowFullScreen
                className="chart-iframe"
              />
            )}
          </div>
        )}

        {/* Error Message Display */}
        {error && (
          <p className="error-message">
            Error: {error}
          </p>
        )}
      </div>
    </div>
  );
};

export default App;
