class Java2 {
    public static void main(String[] args) {
        System.out.println("This is me printing in terminal");


        // strings
        String str1 = new String("This is string 1"); //preferred
        String str2 = "This is string 2";

        System.out.println("Strings declard: " + str1 + " " + str2);
        
        String str = "Java Programming";
        System.out.println(str.length()); // 16
        System.out.println(str.charAt(5)); // 'P'
        System.out.println(str.substring(0, 4)); // "Java"
        System.out.println(str.toUpperCase()); // "JAVA PROGRAMMING"

        String[] parts = str.split(" ");
        for (String word : parts) {
            System.out.println(word);
        }
    }
}