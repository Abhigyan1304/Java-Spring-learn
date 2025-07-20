import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from './store/store';
import { fetchChartHistory, setSelectedChartFromHistory, resetChartState } from './store/chartSlice';

// Define the type for the API response (must match chartSlice)
type ChartApiResponse = {
  sql_query_gen_by_model: string;
  chart_id: string;
  chart_url: string;
  user_id: string;
  chart_type: string;
  timestamp: number;
  question: string;
};

const History: React.FC = () => {
  const dispatch: AppDispatch = useDispatch();
  const { history, loadingHistory, historyError } = useSelector((state: RootState) => state.chart);

  // Fetch history when the component mounts
  useEffect(() => {
    dispatch(fetchChartHistory());
  }, [dispatch]);

  const handleHistoryClick = (chart: ChartApiResponse) => {
    // When a history item is clicked, dispatch an action to set it as the current chart
    dispatch(setSelectedChartFromHistory(chart));
  };

  const handleNewChartClick = () => {
    // Dispatch resetChartState to clear the main display and start a new chart process
    dispatch(resetChartState());
  };

  return (
    <div className="history-container">
      <div className="history-header">
        <h3 className="history-title">Recent History</h3>
        <button className="new-chart-button" onClick={handleNewChartClick}>
          + New Chart
        </button>
      </div>
      {loadingHistory && (
        <p className="history-message">Loading history...</p>
      )}
      {historyError && (
        <p className="history-error">Error: {historyError}</p>
      )}
      {!loadingHistory && history.length === 0 && !historyError && (
        <p className="history-message">No history found. Generate a chart to see it here!</p>
      )}
      <ul className="history-list">
        {history.map((chart) => (
          <li
            key={chart.chart_id}
            className="history-item"
            onClick={() => handleHistoryClick(chart)}
          >
            <p className="history-item-question">
              <strong>Question:</strong> {chart.question}
            </p>
            <p className="history-item-details">
              Type: <strong>{chart.chart_type.toUpperCase()}</strong>
            </p>
            <p className="history-item-details">
              Generated: {new Date(chart.timestamp).toLocaleString()}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default History;
