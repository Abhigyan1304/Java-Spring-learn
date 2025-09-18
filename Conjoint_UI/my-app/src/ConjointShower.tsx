import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from './store';
import './style.css';

const ConjointShower: React.FC = () => {
  const { rows, marketShare } = useSelector((state: RootState) => state.table);

  return (
    <div className="market-container">
      <h2>Market Share</h2>
      <table className="styled-table">
        <thead>
          <tr>
            {rows.map((row, idx) => (
              <th key={idx}>{row.package_name || `Phone ${idx + 1}`}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            {marketShare.map((share, idx) => (
              <td key={idx}>{share}%</td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default ConjointShower;
