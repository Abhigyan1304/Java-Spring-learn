import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

// Define types for the API response
type ChartApiResponse = {
  sql_query_gen_by_model: string;
  chart_id: string;
  chart_url: string;
  user_id: string;
  chart_type: string;
  timestamp: number; // Added for sorting history
  question: string; // Added to display in history
};

// Define types for the state
interface ChartState {
  inputQuestion: string;
  sqlQuery: string | null;
  chartData: ChartApiResponse | null;
  loadingSql: boolean;
  loadingChart: boolean;
  error: string | null;
  sqlContentWriterDone: boolean; // Renamed back: Indicates SQL content typewriter completion
  chartLabelWriterDone: boolean; // To track typewriter completion for chart label
  history: ChartApiResponse[]; // Array to store past charts
  loadingHistory: boolean; // Loading state for history
  historyError: string | null; // Error state for history
}

// Initial state for the slice
const initialState: ChartState = {
  inputQuestion: '',
  sqlQuery: null,
  chartData: null,
  loadingSql: false,
  loadingChart: false,
  error: null,
  sqlContentWriterDone: false, // Initial state for new name
  chartLabelWriterDone: false,
  history: [],
  loadingHistory: false,
  historyError: null,
};

// Async Thunk for SQL generation (simulated API call)
export const generateSqlQuery = createAsyncThunk(
  'chart/generateSqlQuery',
  async (question: string, { rejectWithValue }) => {
    try {
      // Simulate API call to generate SQL with a 1-second delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      const sql = `SELECT * FROM demo_table WHERE query_text = '${question}';`; // Example SQL
      return sql;
    } catch (err: any) {
      return rejectWithValue(err.message || 'Failed to generate SQL query.');
    }
  }
);

// Async Thunk for Chart generation (simulated API call)
export const generateChart = createAsyncThunk(
  'chart/generateChart',
    async ({ sqlQuery, question }: { sqlQuery: string; question: string }, { rejectWithValue }) => {
    try {
        // const chartRes = await fetch("/api/ve");

      // Check if the response was successful (status code 200-299)
    //   if (!chartRes.ok) {
    //     If not OK, try to read the error message from the response body
    //     or default to status text if body is empty/unreadable
    //     const errorText = await chartRes.text();
    //     throw new Error(`HTTP error! Status: ${chartRes.status}, Message: ${errorText || chartRes.statusText}`);
    //   }

        const chartInfo : ChartApiResponse= {
            sql_query_gen_by_model: 'select * from table',
            chart_id: '1',
            chart_url: 'http://youtube.com',
            user_id: '1',
            chart_type: 'youtube chart',
            timestamp: 1,
            question: 'question by me'
        }
      return chartInfo;
    } catch (error: any) {
        return rejectWithValue(error.message || 'Error creating chart.');
    }
    }
);

// New Async Thunk for fetching chart history (simulated API call)
export const fetchChartHistory = createAsyncThunk(
  'chart/fetchChartHistory',
  async (_, { rejectWithValue }) => {
    try {
      await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate 1.5-second delay
      // Generate dummy history data
      const dummyHistory: ChartApiResponse[] = [
        {
          sql_query_gen_by_model: "SELECT * FROM sales_2023;",
          chart_id: "hist_1",
          chart_url: "https://superset.datatest.ch/superset/explore/p/Gj4DoZbRO2R/",
          user_id: "user_1",
          chart_type: "line",
          timestamp: Date.now() - 3600000, // 1 hour ago
          question: "Show me sales data for 2023",
        },
        {
          sql_query_gen_by_model: "SELECT COUNT(*) FROM users WHERE status = 'active';",
          chart_id: "hist_2",
          chart_url: "https://superset.datatest.ch/superset/explore/p/NdoQpn0bDV4/",
          user_id: "user_1",
          chart_type: "pie",
          timestamp: Date.now() - 7200000, // 2 hours ago
          question: "How many active users do we have?",
        },
        {
          sql_query_gen_by_model: "SELECT AVG(price) FROM products WHERE category = 'electronics';",
          chart_id: "hist_3",
          chart_url: "https://superset.datatest.ch/superset/explore/p/NdoQpn0bDV4/",
          user_id: "user_1",
          chart_type: "bar",
          timestamp: Date.now() - 10800000, // 3 hours ago
          question: "Average price of electronics products",
        },
        {
            sql_query_gen_by_model: "SELECT SUM(revenue) FROM daily_reports WHERE date >= '2024-01-01';",
            chart_id: "hist_4",
            chart_url: "https://superset.datatest.ch/superset/explore/p/J4ZDdjepQjP/",
            user_id: "user_1",
            chart_type: "area",
            timestamp: Date.now() - 14400000, // 4 hours ago
            question: "Total revenue since January 2024",
        },
        {
            sql_query_gen_by_model: "SELECT product_name, stock_level FROM inventory ORDER BY stock_level ASC LIMIT 5;",
            chart_id: "hist_5",
            chart_url: "https://superset.datatest.ch/superset/explore/p/j52DbrnEDXo/",
            user_id: "user_1",
            chart_type: "table",
            timestamp: Date.now() - 18000000, // 5 hours ago
            question: "Top 5 lowest stock products",
        },
        {
            sql_query_gen_by_model: "SELECT region, COUNT(DISTINCT customer_id) FROM orders GROUP BY region;",
            chart_id: "hist_6",
            chart_url: "https://superset.datatest.ch/superset/explore/p/gk4DmyLyDBw/",
            user_id: "user_1",
            chart_type: "pie",
            timestamp: Date.now() - 21600000, // 6 hours ago
            question: "Customer count by region",
        },
        {
            sql_query_gen_by_model: "SELECT employee_id, hours_worked FROM timesheets WHERE week = '2024-W28';",
            chart_id: "hist_7",
            chart_url: "https://superset.datatest.ch/superset/explore/p/g7wxlm50Qz0/",
            user_id: "user_1",
            chart_type: "bar",
            timestamp: Date.now() - 25200000, // 7 hours ago
            question: "Employee hours for current week",
        },
        {
            sql_query_gen_by_model: "SELECT event_name, COUNT(*) FROM website_logs GROUP BY event_name ORDER BY COUNT(*) DESC LIMIT 10;",
            chart_id: "hist_8",
            chart_url: "https://superset.datatest.ch/superset/explore/p/17jDZ07mQwY/",
            user_id: "user_1",
            chart_type: "table",
            timestamp: Date.now() - 28800000, // 8 hours ago
            question: "Top 10 website events",
        },
        {
            sql_query_gen_by_model: "SELECT date, daily_visitors FROM traffic_data ORDER BY date DESC LIMIT 10;",
            chart_id: "hist_9",
            chart_url: "https://superset.datatest.ch/superset/explore/p/dpYOVnVPOBv/",
            user_id: "user_1",
            chart_type: "line",
            timestamp: Date.now() - 32400000, // 9 hours ago
            question: "Last 10 days of website visitors",
        },
        {
            sql_query_gen_by_model: "SELECT department, AVG(salary) FROM employees GROUP BY department;",
            chart_id: "hist_10",
            chart_url: "https://superset.datatest.ch/superset/explore/p/l5pO0j12xBo/",
            user_id: "user_1",
            chart_type: "bar",
            timestamp: Date.now() - 36000000, // 10 hours ago
            question: "Average salary by department",
        },
      ].sort((a, b) => b.timestamp - a.timestamp); // Sort by most recent first

      return dummyHistory;
    } catch (err: any) {
      return rejectWithValue(err.message || 'Failed to fetch chart history.');
    }
  }
);

// Create the Redux slice
const chartSlice = createSlice({
  name: 'chart',
  initialState,
  reducers: {
    // Action to set the input question
    setInputQuestion: (state, action: PayloadAction<string>) => {
      state.inputQuestion = action.payload;
    },
    // Action to reset all states before a new submission
    resetChartState: (state) => {
      state.inputQuestion = ''; // Clear input field
      state.sqlQuery = null;
      state.chartData = null;
      state.loadingSql = false;
      state.loadingChart = false;
      state.error = null;
      state.sqlContentWriterDone = false; // Reset for new process
      state.chartLabelWriterDone = false; // Reset chart label typewriter
      // Do NOT reset history here, as it's persistent
    },
    // Action to mark SQL content typewriter as done
    setSqlContentWriterDone: (state, action: PayloadAction<boolean>) => {
      state.sqlContentWriterDone = action.payload;
    },
    // Action to mark chart label typewriter as done
    setChartLabelWriterDone: (state, action: PayloadAction<boolean>) => {
      state.chartLabelWriterDone = action.payload;
    },
    // Action to add a newly generated chart to history
    addChartToHistory: (state, action: PayloadAction<ChartApiResponse>) => {
      // Add to the beginning of the array and keep only the latest 10
      state.history = [action.payload, ...state.history].slice(0, 10);
    },
    // New action to set the main display based on a history item click
    setSelectedChartFromHistory: (state, action: PayloadAction<ChartApiResponse>) => {
      state.sqlQuery = action.payload.sql_query_gen_by_model;
      state.chartData = action.payload;
      state.inputQuestion = action.payload.question; // Update input field with original question
      state.loadingSql = false;
      state.loadingChart = false;
      state.error = null;
      state.sqlContentWriterDone = true; // Assume SQL is "done" if loaded from history
      state.chartLabelWriterDone = true; // Assume chart label is "done" if loaded from history
    },
    // New action to load application state from a parsed ChartApiResponse (e.g., from URL)
    loadAppStateFromEncodedUrl: (state, action: PayloadAction<ChartApiResponse>) => {
      state.sqlQuery = action.payload.sql_query_gen_by_model;
      state.chartData = action.payload;
      state.inputQuestion = action.payload.question;
      state.loadingSql = false;
      state.loadingChart = false;
      state.error = null;
      state.sqlContentWriterDone = true; // Content is already available
      state.chartLabelWriterDone = true; // Label is already available
    },
  },
  extraReducers: (builder) => {
    builder
      // Reducers for generateSqlQuery thunk
      .addCase(generateSqlQuery.pending, (state) => {
        state.loadingSql = true;
        state.error = null; // Clear previous errors
      })
      .addCase(generateSqlQuery.fulfilled, (state, action: PayloadAction<string>) => {
        state.loadingSql = false;
        state.sqlQuery = action.payload;
        state.sqlContentWriterDone = false; // Reset for typewriter effect on SQL content
      })
      .addCase(generateSqlQuery.rejected, (state, action: PayloadAction<any>) => {
        state.loadingSql = false;
        state.error = action.payload;
      })
      // Reducers for generateChart thunk
      .addCase(generateChart.pending, (state) => {
        state.loadingChart = true;
        state.error = null; // Clear previous errors
        state.chartLabelWriterDone = false; // Reset for new chart generation
      })
      .addCase(generateChart.fulfilled, (state, action: PayloadAction<ChartApiResponse>) => {
        state.loadingChart = false;
        state.chartData = action.payload;
        // Correctly pass the payload to addChartToHistory
        chartSlice.caseReducers.addChartToHistory(state, { payload: action.payload, type: 'chart/addChartToHistory' });
      })
      .addCase(generateChart.rejected, (state, action: PayloadAction<any>) => {
        state.loadingChart = false;
        // Set a specific error message for chart generation failure
        state.error = action.payload || 'Error creating chart.';
      })
      // Reducers for fetchChartHistory thunk (new)
      .addCase(fetchChartHistory.pending, (state) => {
        state.loadingHistory = true;
        state.historyError = null;
      })
      .addCase(fetchChartHistory.fulfilled, (state, action: PayloadAction<ChartApiResponse[]>) => {
        state.loadingHistory = false;
        state.history = action.payload;
      })
      .addCase(fetchChartHistory.rejected, (state, action: PayloadAction<any>) => {
        state.loadingHistory = false;
        state.historyError = action.payload;
      });
  },
});

export const {
  setInputQuestion,
  resetChartState,
  setSqlContentWriterDone,
  setChartLabelWriterDone,
  addChartToHistory,
  setSelectedChartFromHistory,
  loadAppStateFromEncodedUrl,
} = chartSlice.actions;

export default chartSlice.reducer;
