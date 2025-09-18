import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from './store';
import { updateCell, addPhoneColumn } from './tableSlice';
import './style.css';

// Tooltip component
const Tooltip: React.FC<{ message: string }> = ({ message, children }) => {
  const [visible, setVisible] = useState(false);
  return (
    <span
      style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <span
          style={{
            position: 'absolute',
            bottom: '120%',
            left: '50%',
            transform: 'translateX(-50%)',
            background: '#333',
            color: '#fff',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            whiteSpace: 'nowrap',
            zIndex: 10,
          }}
        >
          {message}
        </span>
      )}
    </span>
  );
};

const INITIAL_PHONE = {
  package_name: "New Model",
  brand: "BrandNew",
  processor: "Snapdragon",
  screen_size: "6.1\"",
  screen_type: "AMOLED",
  resolution: "FHD+",
  ram: "8GB",
  rear_camera: "50MP",
  front_camera: "16MP",
  side_frame: "Aluminum",
  back_material: "Glass",
  battery_capacity: "4500mAh",
  charging_speed: "67W",
  inbox_charger: "Available",
  os_update_policy: "3 Years",
  ip_protection: "IP68",
  price: 29999
};

const MIN_PRICE = 29000;
const MAX_PRICE = 34999;

const TableGrid: React.FC = () => {
  const { schema, rows } = useSelector((state: RootState) => state.table);
  const dispatch = useDispatch<AppDispatch>();

  const handleAddColumn = () => {
    dispatch(addPhoneColumn(INITIAL_PHONE));
  };

  const handlePriceBlur = (rowIndex: number, colId: string, value: string) => {
    let num = parseInt(value.replace(/\D/g, ""), 10);
    if (isNaN(num)) num = MIN_PRICE;
    if (num < MIN_PRICE) num = MIN_PRICE;
    if (num > MAX_PRICE) num = MAX_PRICE;
    dispatch(updateCell({ rowIndex, colId, value: num }));
  };

  return (
    <div className="table-container">
      <table className="styled-table">
        <thead>
          <tr>
            <th>Model Name</th>
            {rows.map((row, rowIndex) => (
              <th key={rowIndex}>{row.package_name || `Phone ${rowIndex + 1}`}</th>
            ))}
            <th>
              <button onClick={handleAddColumn} title="Add Phone Column">＋</button>
            </th>
          </tr>
        </thead>
        <tbody>
          {schema.map((col) => (
            <tr key={col.id}>
              <td>{col.label}</td>
              {rows.map((row, rowIndex) => (
                <td key={`${col.id}-${rowIndex}`}>
                  {col.id === "price" ? (
                    <Tooltip message={`Value must be between ${MIN_PRICE} and ${MAX_PRICE}`}>
                      <input
                        type="text"
                        value={row[col.id] ?? ""}
                        onChange={e =>
                          dispatch(updateCell({ rowIndex, colId: col.id, value: e.target.value }))
                        }
                        onBlur={e =>
                          handlePriceBlur(rowIndex, col.id, e.target.value)
                        }
                        title={`Value must be between ${MIN_PRICE} and ${MAX_PRICE}`}
                      />
                    </Tooltip>
                  ) : col.type === "dropdown" ? (
                    <select
                      value={row[col.id] as string}
                      onChange={(e) =>
                        dispatch(updateCell({ rowIndex, colId: col.id, value: e.target.value }))
                      }
                    >
                      {col.options?.map((opt: string) => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={row[col.id] as string}
                      onChange={(e) =>
                        dispatch(updateCell({ rowIndex, colId: col.id, value: e.target.value }))
                      }
                    />
                  )}
                </td>
              ))}
              <td></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TableGrid;