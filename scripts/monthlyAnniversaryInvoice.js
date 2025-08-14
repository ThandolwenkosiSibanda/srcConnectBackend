import { createClient } from "@supabase/supabase-js";

// Load from environment or config
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// Helper function to get today's day (e.g., 21 for July 21)
const getTodayDay = () => new Date().getDate();

async function generateMonthlyAnniversaryInvoices() {
  const todayDay = getTodayDay();

  console.log(`Running monthly invoice job for day: ${todayDay}`);

  // Step 1: Get all active policies whose start_date matches today's day
  const { data: policies, error: policyError } = await supabase
    .from("policies")
    .select("*")
    .eq("status", "active");

  if (policyError) {
    console.error("Error fetching policies: now", policyError);
    return;
  }

  const duePolicies = policies.filter((policy) => {
    const startDate = new Date(policy.start_date);
    return startDate.getDate() === todayDay;
  });

  console.log(`Found ${duePolicies.length} policies due today.`);

  // Step 2: Generate an invoice for each due policy
  //   for (const policy of duePolicies) {
  //     const invoicePayload = {
  //       policy_id: policy.id,
  //       client_id: policy.client_id,
  //       amount: policy.monthly_premium,
  //       currency: policy.currency,
  //       status: "unpaid",
  //       issued_at: new Date().toISOString(),
  //     };

  //     const { error: invoiceError } = await supabase
  //       .from("invoices")
  //       .insert([invoicePayload]);

  //     if (invoiceError) {
  //       console.error(
  //         `Failed to create invoice for policy ${policy.id}:`,
  //         invoiceError
  //       );
  //     } else {
  //       console.log(`Invoice created for policy ${policy.id}`);
  //     }
  //   }

  for (const policy of duePolicies) {
    const invoicePayload = {
      policy_id: policy.id,
      client_id: policy.client_id,
      amount: policy.monthly_premium,
      currency: policy.currency,
      status: "unpaid",
      issued_at: new Date().toISOString(),
    };

    // Insert invoice
    const { data: invoiceData, error: invoiceError } = await supabase
      .from("invoices")
      .insert([invoicePayload])
      .select()
      .single();

    if (invoiceError) {
      console.error(
        `❌ Failed to create invoice for policy ${policy.id}:`,
        invoiceError
      );
      continue;
    } else {
      console.log(`✅ Invoice created for policy ${policy.id}`);
    }

    // Fetch client's wallet
    const { data: walletData, error: walletError } = await supabase
      .from("wallets")
      .select("balance")
      .eq("client_id", policy.client_id)
      .single();

    if (walletError) {
      console.error(
        `⚠️ Failed to fetch wallet for client ${policy.client_id}:`,
        walletError
      );
      continue;
    }

    const balance = parseFloat(walletData?.balance || 0);
    const invoiceAmount = parseFloat(invoicePayload.amount);

    if (balance >= invoiceAmount) {
      // Deduct wallet balance
      const { error: walletUpdateError } = await supabase
        .from("wallets")
        .update({
          balance: balance - invoiceAmount,
          last_updated: new Date().toISOString(),
        })
        .eq("client_id", policy.client_id);

      if (walletUpdateError) {
        console.error(
          `⚠️ Failed to deduct balance for client ${policy.client_id}:`,
          walletUpdateError
        );
        continue;
      }

      // Mark invoice as paid
      const { error: invoiceUpdateError } = await supabase
        .from("invoices")
        .update({
          status: "paid",
          paid_at: new Date().toISOString(),
        })
        .eq("id", invoiceData.id);

      if (invoiceUpdateError) {
        console.error(
          `⚠️ Failed to update invoice ${invoiceData.id} as paid:`,
          invoiceUpdateError
        );
        continue;
      }

      // Log transaction (optional)
      await supabase.from("wallet_transactions").insert([
        {
          client_id: policy.client_id,
          amount: -invoiceAmount,
          type: "debit",
          invoice_id: invoiceData.id,
          created_at: new Date().toISOString(),
        },
      ]);

      console.log(
        `💰 Invoice ${invoiceData.id} auto-paid for client ${policy.client_id}`
      );
    } else {
      console.log(
        `⛔ Not enough balance in wallet for client ${policy.client_id}. Invoice remains unpaid.`
      );
    }
  }

  console.log("Monthly anniversary invoice job completed.");
}

// Run the script if executed directly (node monthlyAnniversaryInvoice.js)
if (import.meta.url === `file://${process.argv[1]}`) {
  generateMonthlyAnniversaryInvoices().catch(console.error);
}
