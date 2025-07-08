import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import javax.naming.spi.DirStateFactory.Result;

public class DBFunctions {

    private static final String URL = "jdbc:postgresql://localhost:5432/Demo";
    private static final String USER = "postgres";
    private static final String PASSWORD = "Disha@2402";

    public static Connection getConnection(String URL, String USER, String PASSWORD) {
        Connection connection = null;
        try {
            Class.forName("org.postgresql.Driver");
            connection = DriverManager.getConnection(URL, USER, PASSWORD);
            System.out.println("CONNECTION DONE!");
        } catch (ClassNotFoundException | SQLException e) {
            System.out.println("Exception accured");
            e.printStackTrace();
        }
        return connection;
    }

    public static void closeConnection(Connection connection) {
        if (connection != null) {
            try {
                connection.close();
                System.out.println("Connection closed successfully.");
            } catch (SQLException e) {
                System.out.println("Error closing connection.");
                e.printStackTrace();
            }
        }
    }

    public static void createTable(Connection connection, String tableName) {
        String createTableSQL = "CREATE TABLE IF NOT EXISTS " + tableName + "("
                + "id SERIAL PRIMARY KEY, "
                + "name VARCHAR(100) NOT NULL, "
                + "email VARCHAR(100) NOT NULL UNIQUE"
                + ");";
        try {
            connection.createStatement().execute(createTableSQL);
            System.out.println("Table created successfully.");
        } catch (SQLException e) {
            System.out.println("Error creating table.");
            e.printStackTrace();
        }
    }

    public static void insertRow(Connection conn, String tableName, String name, String email) {
        // String insertSQL = "INSERT INTO " + tableName + " (name, email) VALUES (?,
        // ?)";
        // try (PreparedStatement pstmt = conn.prepareStatement(insertSQL)) {
        // pstmt.setString(1, name);
        // pstmt.setString(2, email);
        // pstmt.executeUpdate();
        // System.out.println("Row inserted successfully.");
        Statement statement;
        try {
            String query = String.format("Insert into %s(name,email) values ('%s' , '%s')", tableName, name, email);
            statement = conn.createStatement();
            statement.executeUpdate(query);
            System.out.println("Row inserted successfully.");
        } catch (Exception e) {
            System.out.println("Error inserting row.");
            e.printStackTrace();
        }
    }

    public static void readData(Connection conn, String tableName) {
        Statement statement;
        ResultSet resultSet;
        try {
            String query = String.format("SELECT * FROM %s", tableName);
            statement = conn.createStatement();
            resultSet = statement.executeQuery(query);
            while (resultSet.next()) {
                int id = resultSet.getInt("id");
                String name = resultSet.getString("name");
                String email = resultSet.getString("email");
                System.out.println("ID: " + id + ", Name: " + name + ", Email: " + email);
            }
            System.out.println(resultSet);
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void updateData(Connection conn, String tableName, String name, String newEmail) {
        Statement statement = null;
        try {
            String query = String.format("UPDATE %s SET email = '%s' where name = '%s'", tableName, newEmail, name);
            statement = conn.createStatement();
            statement.executeUpdate(query);
            System.out.println("Data updated successfully.");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void deleteData(Connection conn, String tableName, String name) {
        Statement statement = null;
        try {
            String query = String.format("DELETE FROM %s WHERE name = '%s'", tableName, name);
            statement = conn.createStatement();
            statement.executeUpdate(query);
            System.out.println("Data deleted successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
