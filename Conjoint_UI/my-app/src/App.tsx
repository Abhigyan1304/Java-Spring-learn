import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { setSchema, setRows } from './tableSlice';
import { AppDispatch } from './store';
import schema from './schema.json';
import rows from './data.json';
import TableGrid from './TableGrid';
import ConjointShower from './ConjointShower';
import './style.css';

const App: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();

  useEffect(() => {
    dispatch(
      setSchema(
        schema.map((col: any) => ({
          ...col,
          type: col.type as "text" | "dropdown"
        }))
      )
    );
    dispatch(setRows(rows));
  }, [dispatch]);

  return (
    <div>
      <h1>Excel-like Table</h1>
      <TableGrid />
      <ConjointShower />
    </div>
  );
};

export default App;
