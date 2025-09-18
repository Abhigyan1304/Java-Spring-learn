import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface ColumnSchema {
  id: string;
  label: string;
  type: 'text' | 'dropdown' | 'number';
  options?: string[];
}

export interface Row {
  [key: string]: string | boolean | number;
}

interface TableState {
  columns: any;
  schema: ColumnSchema[];
  rows: Row[];
}

const initialState: TableState = {
  columns: [],
  schema: [],
  rows: [],
};

const tableSlice = createSlice({
  name: 'table',
  initialState,
  reducers: {
    setSchema(state, action: PayloadAction<ColumnSchema[]>) {
      state.schema = action.payload;
    },
    setRows(state, action: PayloadAction<Row[]>) {
      // Remove $ from price and convert to number
      state.rows = action.payload.map(row => ({
        ...row,
        price: typeof row.price === "string"
          ? parseInt(row.price.replace(/\D/g, ""), 10)
          : row.price
      }));
    },
    updateCell(
      state,
      action: PayloadAction<{ rowIndex: number; colId: string; value: string | boolean | number }>
    ) {
      const { rowIndex, colId, value } = action.payload;
      state.rows[rowIndex][colId] = value;
    },
    addPhoneColumn(state, action: PayloadAction<Row>) {
      state.rows.push({ ...action.payload });
    }
  },
});

export const { setSchema, setRows, updateCell, addPhoneColumn } = tableSlice.actions;
export default tableSlice.reducer;
