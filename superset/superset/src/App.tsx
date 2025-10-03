import React, { useState, useEffect, useRef } from 'react';
import { configureStore, createSlice, createAsyncThunk, type PayloadAction } from '@reduxjs/toolkit';
import { Provider, useSelector, useDispatch } from 'react-redux';
import './App.css'; // Updated CSS file name

// --- 1. TYPE DEFINITIONS ---
// Define the types for the data objects and the state.
interface ChartData {
  chart_id: string;
  dataset_id: string;
  chart_type: string;
  iframeUrl: string;
  user_id: string;
  user_input: string;
}

interface AppState {
  isSqlLoading: boolean;
  isChartLoading: boolean;
  // Separate loading states for suggestions and summary
  isSuggestionsLoading: boolean;
  isSummaryLoading: boolean;
  sqlError: string | null;
  chartError: string | null;
  insightsError: string | null;
  sqlQuery: string | null;
  chartData: ChartData | null;
  summary: string | null;
  suggestions: string[] | null;
}

// --- 2. REDUX STATE MANAGEMENT ---

// Define the initial state for our Redux slice
const initialState: AppState = {
  isSqlLoading: false,
  isChartLoading: false,
  isSuggestionsLoading: false,
  isSummaryLoading: false,
  sqlError: null,
  chartError: null,
  insightsError: null,
  sqlQuery: null,
  chartData: null,
  summary: null,
  suggestions: null,
};

// Create a Redux slice with the reducer logic
const dataSlice = createSlice({
  name: 'data',
  initialState,
  reducers: {
    resetState: (state) => {
      state.sqlError = null;
      state.chartError = null;
      state.insightsError = null;
      state.sqlQuery = null;
      state.chartData = null;
      state.summary = null;
      state.suggestions = null;
      state.isSuggestionsLoading = false;
      state.isSummaryLoading = false;
    },
    setSqlError: (state, action: PayloadAction<string | null>) => {
      state.sqlError = action.payload;
    },
    setChartError: (state, action: PayloadAction<string | null>) => {
      state.chartError = action.payload;
    },
    setInsightsError: (state, action: PayloadAction<string | null>) => {
      state.insightsError = action.payload;
    },
    setSuggestions: (state, action: PayloadAction<string[]>) => {
      state.suggestions = action.payload;
    },
    setSummary: (state, action: PayloadAction<string>) => {
      state.summary = action.payload;
    },
    // New reducers for the separate loading states
    setSuggestionsLoading: (state, action: PayloadAction<boolean>) => {
      state.isSuggestionsLoading = action.payload;
    },
    setSummaryLoading: (state, action: PayloadAction<boolean>) => {
      state.isSummaryLoading = action.payload;
    },
    // New reducer to handle setting chart data from a decoded URL
    setChartDataFromUrl: (state, action: PayloadAction<ChartData | null>) => {
      state.chartData = action.payload;
      if (action.payload) {
        // Here's the key fix: we set the sqlQuery state from the user_input
        // property in the ChartData object that's decoded from the URL.
        state.sqlQuery = action.payload.user_input;
        state.isSqlLoading = false;
        state.isChartLoading = false;
        state.isSuggestionsLoading = false;
        state.isSummaryLoading = false;
      }
    },
  },
  extraReducers: (builder) => {
    // Reducers for the SQL query thunk
    builder
      .addCase(generateSqlQueryThunk.pending, (state) => {
        state.isSqlLoading = true;
        state.sqlError = null;
        state.sqlQuery = null;
        state.chartData = null;
        state.summary = null;
        state.suggestions = null;
        state.isSuggestionsLoading = false;
        state.isSummaryLoading = false;
      })
      .addCase(generateSqlQueryThunk.fulfilled, (state, action: PayloadAction<string>) => {
        state.isSqlLoading = false;
        state.sqlQuery = action.payload;
      })
      .addCase(generateSqlQueryThunk.rejected, (state, action) => {
        state.isSqlLoading = false;
        state.sqlError = action.error.message || 'Failed to generate SQL query.';
      });

    // Reducers for the chart data thunk
    builder
      .addCase(generateChartDataThunk.pending, (state) => {
        state.isChartLoading = true;
        state.chartError = null;
        state.chartData = null;
        state.summary = null;
        state.suggestions = null;
      })
      .addCase(generateChartDataThunk.fulfilled, (state, action: PayloadAction<ChartData>) => {
        state.isChartLoading = false;
        state.chartData = action.payload;
        // After receiving chart data, update the URL
        const encodedData = btoa(JSON.stringify(action.payload));
        window.history.pushState({}, '', `?data=${encodedData}`);
      })
      .addCase(generateChartDataThunk.rejected, (state, action) => {
        state.isChartLoading = false;
        state.chartError = action.error.message || 'Failed to generate chart data.';
      });
  },
});

export const {
  resetState,
  setSqlError,
  setChartError,
  setInsightsError,
  setSuggestions,
  setSummary,
  setSuggestionsLoading,
  setSummaryLoading,
  setChartDataFromUrl,
} = dataSlice.actions;

// --- 3. ASYNCHRONOUS API LOGIC (Redux Thunk) ---
// Thunk to generate the SQL query (now with a real API call)
export const generateSqlQueryThunk = createAsyncThunk(
  'data/generateSqlQuery',
  async (userPrompt: string, { rejectWithValue }) => {
    try {
      const response = await fetch('http://localhost:8000/api/generate-sql', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_prompt: userPrompt }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        return rejectWithValue(errorData.error || 'Failed to generate SQL query.');
      }
      const data = await response.json();
      return data.sql_query;
    } catch (error) {
      console.error('API call failed:', error);
      return rejectWithValue('Network error or server unavailable.');
    }
  }
);

// Thunk to generate chart data (now with a real API call)
export const generateChartDataThunk = createAsyncThunk(
  'data/generateChartData',
  async (sqlQuery: string, { rejectWithValue }) => {
    // A small comment update to trigger a re-evaluation of the module by Vite
    try {
      const response = await fetch('http://localhost:8000/api/generate-chart-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sql_query: sqlQuery }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        return rejectWithValue(errorData.error || 'Failed to generate chart data.');
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API call failed:', error);
      return rejectWithValue('Network error or server unavailable.');
    }
  }
);

// Configure the Redux store
const store = configureStore({
  reducer: {
    data: dataSlice.reducer,
  },
});

// --- 4. REACT COMPONENTS ---
const DataQueryInterface: React.FC = () => {
  const dispatch = useDispatch();
  const { isSqlLoading, isChartLoading, isSuggestionsLoading, isSummaryLoading, sqlError, chartError, insightsError, sqlQuery, chartData, summary, suggestions } = useSelector((state: { data: AppState }) => state.data);
  const [queryInput, setQueryInput] = useState<string>('');
  const [isSummaryVisible, setIsSummaryVisible] = useState<boolean>(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Effect to handle URL parameters on page load
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const encodedData = params.get('data');
    if (encodedData) {
      try {
        const decodedData = atob(encodedData);
        const parsedData: ChartData = JSON.parse(decodedData);
        dispatch(setChartDataFromUrl(parsedData));
      } catch (e) {
        console.error("Failed to decode or parse chart data from URL:", e);
        dispatch(setChartError("Invalid chart data in URL."));
      }
    }
  }, [dispatch]);

  // Effect to manage the SSE connection lifecycle
  useEffect(() => {
    // Only proceed if chartData exists.
    if (chartData) {
      // Set both loading states to true as we are starting the stream for both suggestions and summary.
      dispatch(setSuggestionsLoading(true));
      dispatch(setSummaryLoading(true));
      dispatch(setInsightsError(null));

      // Close any existing connection before starting a new one
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      eventSourceRef.current = new EventSource("http://localhost:8000/api/v1/sse/insights");

      // Log a message when the connection is successfully opened
      eventSourceRef.current.onopen = () => {
        console.log('SSE connection opened successfully.');
      };

      eventSourceRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.suggestions) {
            dispatch(setSuggestions(data.suggestions));
            dispatch(setSuggestionsLoading(false));
          } else if (data.summary) {
            dispatch(setSummary(data.summary));
            dispatch(setSummaryLoading(false));
          }
        } catch (e) {
          console.error("Failed to parse SSE data:", e);
          dispatch(setInsightsError("Failed to parse server response."));
          dispatch(setSuggestionsLoading(false));
          dispatch(setSummaryLoading(false));
        }
      };

      eventSourceRef.current.onerror = (error) => {
        // Check the readyState to see if it's a normal closure or a real error.
        // readyState 2 means the connection is closed.
        if (eventSourceRef.current && eventSourceRef.current.readyState === EventSource.CLOSED) {
          console.warn('SSE connection was closed, likely by the server. This may be a normal event.');
        } else {
          console.error('SSE Error:', error);
          dispatch(setInsightsError('Failed to load insights via SSE.'));
          dispatch(setSuggestionsLoading(false));
          dispatch(setSummaryLoading(false));
        }
        
        // Always close the connection on error to avoid further events
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
      };
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [chartData, dispatch]);


  // Handle the initial submission and trigger the first thunk
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (queryInput.trim()) {
      dispatch(resetState()); // Clear previous state
      setIsSummaryVisible(false); // Hide summary until button is clicked
      dispatch(generateSqlQueryThunk(queryInput.trim()) as any);
    } else {
      dispatch(setSqlError('Please enter a query.'));
    }
  };

  // Handle clicking on a suggestion bubble
  const handleSuggestionClick = (suggestionText: string) => {
    setQueryInput(suggestionText);
    dispatch(resetState());
    setIsSummaryVisible(false);
    dispatch(generateSqlQueryThunk(suggestionText) as any);
  };

  // Watch for a successful SQL query and trigger the next thunk
  useEffect(() => {
    if (sqlQuery) {
      dispatch(generateChartDataThunk(sqlQuery) as any);
    }
  }, [sqlQuery, dispatch]);

  const handleSummarizeClick = () => {
    setIsSummaryVisible(true);
    // The SSE stream is already open and fetching the summary.
    // The reducer will handle setting isSummaryLoading to false when the summary arrives.
    // We don't need to set the loading state here as it's already set by the useEffect.
    // However, we need to ensure the spinner is shown correctly.
    if (!summary) {
      dispatch(setSummaryLoading(true));
    }
  };

  // New function to handle the reset action
  const handleReset = () => {
    dispatch(resetState());
    setQueryInput('');
  };


  return (
    <div className="container-main">
      <div className="header-content">
        <h1 className="title">Introducing Gemini</h1>
        <p className="subtitle">Gemini now has our smartest, fastest, most useful model yet,<br/>with thinking built in - so you get the best answer, every time.</p>
      </div>

      {/* Input Bar and Button */}
      <form onSubmit={handleSubmit} className="input-container">
        <div className="search-bar">
          <button type="button" onClick={handleReset} id="resetBtn" className="plus-icon-button" disabled={isSqlLoading}>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-rotate-ccw"><path d="M3 12a9 9 0 1 0 9-9h-1A9 9 0 0 0 3 12Z"/><path d="M8 7v5l4-4"/></svg>
          </button>
          <input
            type="text"
            id="queryInput"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="e.g., Show me the total sales by month for the last year."
            className="input-field"
          />
          <div className="mic-icons">
            <button type="submit" className="icon-button" disabled={isSqlLoading}>
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-send-horizonal"><path d="m3 3 3 9-3 9 19-9Z"/><path d="M6 12h16"/></svg>
            </button>
          </div>
        </div>
      </form>

      {/* Output Display Area */}
      <div className="output-container">
        {sqlError && (
          <div className="error-alert" role="alert">
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline ml-2">{sqlError}</span>
          </div>
        )}

        {(sqlQuery || isSqlLoading) && (
          <div className="space-y-8">
            {/* SQL Query Card */}
            <div className="result-card">
              <h2 className="card-title">
                Generated SQL Query
                {isSqlLoading && <div className="loading-animation ml-3"></div>}
              </h2>
              {sqlQuery && <pre className="code-block">{sqlQuery}</pre>}
            </div>

            {/* Chart Card */}
            {chartData || isChartLoading || chartError ? (
              <div className="result-card">
                <h2 className="card-title">
                  Chart Visualization
                  {isChartLoading && <div className="loading-animation ml-3"></div>}
                </h2>
                {chartData && (
                  <>
                    <div className="chart-container">
                      <iframe
                        width="600"
                        height="400"
                        seamless
                        frameBorder="0"
                        scrolling="no"
                        src={chartData.iframeUrl}
                      ></iframe>
                    </div>
                    <div className="chart-info">
                      <p><strong>Chart Type:</strong> {chartData.chart_type}</p>
                      <p><strong>Chart ID:</strong> {chartData.chart_id}</p>
                    </div>
                  </>
                )}
                {chartError && (
                  <div className="error-alert" role="alert">
                    <strong className="font-bold">Error!</strong>
                    <span className="block sm:inline ml-2">{chartError}</span>
                  </div>
                )}
              </div>
            ) : null}

            {/* Summary and Suggestions Card */}
            {chartData && (
              <div className="result-card">
                <div className="card-title-with-button">
                  <h2 className="card-title">
                    Summary & Suggestions
                    {(isSuggestionsLoading || isSummaryLoading) && <div className="loading-animation ml-3"></div>}
                  </h2>
                  <button onClick={handleSummarizeClick} className="summarize-button" disabled={isSummaryLoading || isSummaryVisible}>
                    Summarise
                  </button>
                </div>

                {/* Suggestions are now always visible if they exist */}
                {suggestions && suggestions.length > 0 && (
                  <div className="suggestions-container">
                    {suggestions.map((s, index) => (
                      <div
                        key={index}
                        className="suggestion-bubble"
                        onClick={() => handleSuggestionClick(s)}
                      >
                        {s}
                      </div>
                    ))}
                  </div>
                )}

                {/* Summary is only visible when the button is clicked */}
                {isSummaryVisible && (
                  <div className="mt-4">
                    {isSummaryLoading && !summary ? (
                      <div className="flex items-center justify-center space-x-2">
                        <div className="loading-spinner"></div>
                        <p>Summarizing...</p>
                      </div>
                    ) : summary ? (
                      <p className="summary-text">{summary}</p>
                    ) : null}
                  </div>
                )}

                {/* Display error if there is one */}
                {insightsError && (
                  <div className="error-alert mt-4" role="alert">
                    <strong className="font-bold">Error!</strong>
                    <span className="block sm:inline ml-2">{insightsError}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Main App component to provide the Redux store
const App: React.FC = () => {
  return (
    <Provider store={store}>
      <DataQueryInterface />
    </Provider>
  );
};

export default App;
