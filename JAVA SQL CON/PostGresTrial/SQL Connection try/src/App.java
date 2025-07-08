import java.sql.Connection;

public class App {
    private static final String URL = "jdbc:postgresql://localhost:5432/postgres";
    private static final String USER = "postgres";
    private static final String PASSWORD = "Disha@2402";

    public static void main(String[] args) throws Exception {
        Connection con = null;
        try {
            con = DBFunctions.getConnection(URL, USER, PASSWORD);
            if (con != null) {
                // DBFunctions.createTable(con, "Teachers");
                // DBFunctions.insertRow(con, "Teachers", "Abhigyan Mehta",
                // "abhigyan.m@samsung");
                // DBFunctions.insertRow(con, "Teachers", "Disha Mehta", "disha.m@samsung");
                DBFunctions.updateData(con, "Teachers", "Disha Mehta", "disha@jpmc.com");
                DBFunctions.readData(con, "Teachers");
                DBFunctions.deleteData(con, "Teachers", "Disha Mehta");
                DBFunctions.readData(con, "Teachers");
                DBFunctions.closeConnection(con);
            }
        } catch (Exception e) {
            // TODO: handle exception
            System.out.println("Exception occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
