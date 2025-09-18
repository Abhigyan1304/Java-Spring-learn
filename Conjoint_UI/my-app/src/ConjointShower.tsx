import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from './store';
import './style.css';

const ConjointShower: React.FC = () => {
  const columns = useSelector((state: RootState) => state.table.columns);
  const rowCount = columns.length > 0 ? columns[0].values.length : 0;

  const marketShare = Array.from({ length: rowCount }).map((_, rowIndex) => {
    const counts: Record<string, number> = {};
    columns.forEach((col: { values: any[]; }) => {
      const cell = col.values[rowIndex];
      if (cell.type === "dropdown") {
        const val = cell.value;
        counts[val] = (counts[val] || 0) + 1;
      }
    });
    return counts;
  });

  return (
    <div className="market-container">
      <h2>Market Share (per row)</h2>
      <table className="styled-table">
        <thead>
          <tr>
            <th>Row</th>
            <th>OptionA</th>
            <th>OptionB</th>
            <th>OptionC</th>
          </tr>
        </thead>
        <tbody>
          {marketShare.map((counts, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>{counts["OptionA"] || 0}</td>
              <td>{counts["OptionB"] || 0}</td>
              <td>{counts["OptionC"] || 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ConjointShower;
